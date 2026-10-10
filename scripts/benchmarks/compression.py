from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch

from app import pql
from app.config import SPECTRUM_TOKEN_RANK, TOP_CODES, VOCABULARY
from app.predictions import Mode, coefficients, image_token_codes

KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MASSES = (0.9, 0.99, 0.999)
RANKS = (8, 16, 32, 64, 128, 256)
RESTS = ("spread", "drop")
CODE_BITS = 16
FLOAT_BITS = 32

Stored = dict[str, torch.Tensor]


class Precision(NamedTuple):
    name: str
    bits: int
    domain: str
    scale: str


FLOATS = (
    Precision("fp32", 32, "log", "none"),
    Precision("fp16", 16, "log", "none"),
    Precision("bf16", 16, "log", "none"),
)
VALUES = (
    *FLOATS,
    Precision("q8_log", 8, "log", "slot"),
    Precision("q8_probability", 8, "probability", "slot"),
    Precision("q4_log", 4, "log", "slot"),
    Precision("q4_probability", 4, "probability", "slot"),
)
COEFFICIENTS = (
    *FLOATS,
    Precision("q8_slot", 8, "coefficient", "slot"),
    Precision("q8_component", 8, "coefficient", "component"),
    Precision("q4_slot", 4, "coefficient", "slot"),
    Precision("q4_component", 4, "coefficient", "component"),
)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def kind(mode: Mode) -> str:
    return mode.name.split("_")[-1]


def floor(scheme: object, vocabulary: int) -> float:
    projected = isinstance(scheme, Projection) or (
        isinstance(scheme, Current) and scheme.kind == "spectrum"
    )
    if projected:
        return pql.SPECTRUM_TOKEN_FLOOR * VOCABULARY / vocabulary
    return torch.finfo(torch.float32).tiny


def step(low: torch.Tensor, high: torch.Tensor, bits: int) -> torch.Tensor:
    return (high - low).clamp_min(torch.finfo(torch.float32).tiny) / (2**bits - 1)


def quantise(
    values: torch.Tensor, bits: int, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    clipped = torch.minimum(torch.maximum(values, low), high)
    return torch.round((clipped - low) / step(low, high, bits)).to(torch.uint8)


def store(values: torch.Tensor, precision: Precision) -> Stored:
    if precision.scale == "none":
        return {"values": values.to(DTYPES[precision.name])}
    low = values.amin(dim=-1, keepdim=True)
    high = values.amax(dim=-1, keepdim=True)
    return {
        "codes": quantise(values, precision.bits, low, high),
        "offsets": low,
        "steps": step(low, high, precision.bits),
    }


def load(stored: Stored) -> torch.Tensor:
    if "values" in stored:
        return stored["values"].float()
    return stored["offsets"] + stored["codes"] * stored["steps"]


def stored_bits(stored: Stored, precision: Precision) -> int:
    if "values" in stored:
        return stored["values"].numel() * precision.bits
    scales = stored["offsets"].numel() + stored["steps"].numel()
    return stored["codes"].numel() * precision.bits + scales * FLOAT_BITS


class Dense:
    def __init__(self, precision: Precision) -> None:
        self.precision = precision
        self.name = f"dense/{precision.name}"

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        if self.precision.domain == "probability":
            return store(log_probabilities.float().exp(), self.precision)
        return store(log_probabilities.float(), self.precision)

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        values = load(stored)
        return values if self.precision.domain == "probability" else values.exp()

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        return (self.decode(stored) * query).sum(dim=-1)

    def bits(self, stored: Stored) -> int:
        return stored_bits(stored, self.precision)


class Kept:
    def __init__(self, rest: str, precision: Precision, vocabulary: int) -> None:
        self.rest = rest
        self.precision = precision
        self.vocabulary = vocabulary

    def kept(
        self, log_probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        log_probabilities = log_probabilities.float()
        indices, mask = self.kept(log_probabilities)
        values = log_probabilities.gather(-1, indices)
        mass = torch.where(mask, values.double().exp(), 0).sum(dim=-1)
        if self.precision.domain == "probability":
            values = values.exp()
        values = torch.where(mask, values, values[..., :1])
        stored = {"indices": indices, "mask": mask}
        for name, array in store(values, self.precision).items():
            stored[f"kept_{name}"] = array
        if self.rest == "spread":
            stored["rest"] = (1 - mass).clamp(0, 1).float()
        return stored

    def parts(
        self, stored: Stored
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = load(
            {
                name.removeprefix("kept_"): array
                for name, array in stored.items()
                if name.startswith("kept_")
            }
        )
        if self.precision.domain == "log":
            values = values.exp()
        mask = stored["mask"]
        probabilities = torch.where(mask, values, 0)
        if self.rest == "drop":
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            spread = torch.zeros(mask.shape[:-1], device=mask.device)
        else:
            outside = (self.vocabulary - mask.sum(dim=-1)).clamp_min(1)
            spread = stored["rest"] / outside
        return stored["indices"], mask, probabilities, spread

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        indices, mask, probabilities, spread = self.parts(stored)
        dense = spread[..., None].expand(*spread.shape, self.vocabulary).clone()
        kept = torch.where(mask, probabilities, spread[..., None])
        return dense.scatter_(-1, indices, kept)

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        indices, mask, probabilities, spread = self.parts(stored)
        at = query.expand(*indices.shape[:-1], query.shape[-1]).gather(-1, indices)
        at = torch.where(mask, at, 0)
        outside = query.sum(dim=-1) - at.sum(dim=-1)
        return (probabilities * at).sum(dim=-1) + spread * outside

    def bits(self, stored: Stored) -> int:
        kept = int(stored["mask"].sum()) * (CODE_BITS + self.precision.bits)
        scales = sum(
            stored[name].numel()
            for name in ("kept_offsets", "kept_steps")
            if name in stored
        )
        rest = stored["rest"].numel() if "rest" in stored else 0
        return kept + (scales + rest) * FLOAT_BITS + self.lengths(stored)

    def lengths(self, stored: Stored) -> int:
        return 0


class TopK(Kept):
    def __init__(
        self, k: int, rest: str, precision: Precision, vocabulary: int
    ) -> None:
        super().__init__(rest, precision, vocabulary)
        self.k = k
        self.name = f"top{k}/{rest}/{precision.name}"

    def kept(
        self, log_probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = log_probabilities.topk(self.k, dim=-1).indices
        return indices, torch.ones_like(indices, dtype=torch.bool)


class TopP(Kept):
    def __init__(
        self, mass: float, rest: str, precision: Precision, vocabulary: int
    ) -> None:
        super().__init__(rest, precision, vocabulary)
        self.mass = mass
        self.name = f"topp{mass}/{rest}/{precision.name}"

    def kept(
        self, log_probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ordered, order = log_probabilities.sort(dim=-1, descending=True)
        cumulative = ordered.double().exp().cumsum(dim=-1)
        counts = ((cumulative < self.mass).sum(dim=-1) + 1).clamp_max(
            log_probabilities.shape[-1]
        )
        width = int(counts.max())
        mask = torch.arange(width, device=counts.device) < counts[..., None]
        return order[..., :width], mask

    def lengths(self, stored: Stored) -> int:
        return stored["mask"][..., 0].numel() * CODE_BITS


class Basis:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def fit(self, chunks: Iterable[torch.Tensor]) -> None:
        rows = [chunk.reshape(-1, chunk.shape[-1]) for chunk in chunks]
        count = sum(len(chunk) for chunk in rows)
        total, products = 0, 0
        for chunk in rows:
            probabilities = chunk.double().exp()
            total = total + probabilities.sum(dim=0)
            products = products + probabilities.T @ probabilities
        mean = total / count
        _, vectors = torch.linalg.eigh(products / count - torch.outer(mean, mean))
        self.mean = mean.float()
        self.directions = vectors.flip(-1)[:, : self.rank].T.float().contiguous()
        projected = [
            (chunk.float().exp() - self.mean) @ self.directions.T for chunk in rows
        ]
        self.low = torch.stack([part.amin(dim=0) for part in projected]).amin(dim=0)
        self.high = torch.stack([part.amax(dim=0) for part in projected]).amax(dim=0)

    def to(self, device: torch.device) -> "Basis":
        moved = Basis(self.rank)
        for name in ("mean", "directions", "low", "high"):
            setattr(moved, name, getattr(self, name).to(device))
        return moved


class Projection:
    def __init__(self, rank: int, precision: Precision, basis: Basis) -> None:
        self.rank = rank
        self.precision = precision
        self.basis = basis
        self.name = f"pca{rank}/{precision.name}"

    def directions(self) -> torch.Tensor:
        return self.basis.directions[: self.rank]

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis.low[: self.rank], self.basis.high[: self.rank]

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        centred = log_probabilities.float().exp() - self.basis.mean
        projected = centred @ self.directions().T
        if self.precision.scale == "component":
            return {"codes": quantise(projected, self.precision.bits, *self.bounds())}
        return store(projected, self.precision)

    def coefficients(self, stored: Stored) -> torch.Tensor:
        if self.precision.scale == "component":
            low, high = self.bounds()
            return low + stored["codes"] * step(low, high, self.precision.bits)
        return load(stored)

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        return self.basis.mean + self.coefficients(stored) @ self.directions()

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        projected_query = query @ self.directions().T
        return query @ self.basis.mean + (
            self.coefficients(stored) * projected_query
        ).sum(dim=-1)

    def bits(self, stored: Stored) -> int:
        if self.precision.scale == "component":
            return stored["codes"].numel() * self.precision.bits
        return stored_bits(stored, self.precision)


class Current:
    def __init__(self, mode: Mode, basis: Basis) -> None:
        self.vocabulary = mode.vocabulary
        self.kind = kind(mode)
        self.basis = basis
        self.dense = Dense(FLOATS[1])
        self.name = "current"

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        if self.kind not in ("image", "spectrum"):
            return self.dense.encode(log_probabilities)
        shape = log_probabilities.shape[:-1]
        flat = log_probabilities.reshape(-1, log_probabilities.shape[-1])
        flat = flat.float().cpu().numpy()
        if self.kind == "image":
            parts = image_token_codes(flat)
            names = ("codes", "kept", "tails")
        else:
            parts = coefficients(
                np.exp(flat),
                self.basis.mean.cpu().numpy(),
                self.basis.directions[:SPECTRUM_TOKEN_RANK].cpu().numpy(),
            )
            names = ("quantised", "offsets", "steps")
        return {
            name: torch.from_numpy(array)
            .reshape(*shape, *array.shape[1:])
            .to(log_probabilities.device)
            for name, array in zip(names, parts, strict=True)
        }

    def image(
        self, stored: Stored, kept: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = stored["kept"].float().exp()
        rest = stored["tails"].exp() + probabilities[..., kept:].sum(dim=-1)
        return (
            stored["codes"][..., :kept].long(),
            probabilities[..., :kept],
            rest / (self.vocabulary - kept),
        )

    def projected(self, stored: Stored) -> torch.Tensor:
        return (
            stored["offsets"][..., None]
            + stored["quantised"] * stored["steps"][..., None]
        )

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        if self.kind == "image":
            kept = pql.KEPT if side == "gallery" else TOP_CODES
            codes, probabilities, spread = self.image(stored, kept)
            dense = spread[..., None].expand(*spread.shape, self.vocabulary).clone()
            return dense.scatter_(-1, codes, probabilities)
        if self.kind == "spectrum":
            directions = self.basis.directions[:SPECTRUM_TOKEN_RANK]
            return self.basis.mean + self.projected(stored) @ directions
        return self.dense.decode(stored)

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        if self.kind == "image":
            parts = [part.cpu().numpy() for part in self.image(stored, pql.KEPT)]
            overlaps = pql.image_token_overlaps(query.cpu().numpy(), *parts)
            return torch.from_numpy(overlaps).to(query.device)
        if self.kind == "spectrum":
            directions = self.basis.directions[:SPECTRUM_TOKEN_RANK]
            return query @ self.basis.mean + (
                self.projected(stored) * (query @ directions.T)
            ).sum(dim=-1)
        return self.dense.overlap(stored, query)

    def bits(self, stored: Stored) -> int:
        return 8 * sum(
            array.numel() * array.element_size() for array in stored.values()
        )


def schemes(mode: Mode, basis: Basis) -> list:
    vocabulary = mode.vocabulary
    return [
        Current(mode, basis),
        *(Dense(precision) for precision in VALUES),
        *(
            TopK(k, rest, precision, vocabulary)
            for k in KS
            for rest in RESTS
            for precision in VALUES
        ),
        *(
            TopP(mass, rest, precision, vocabulary)
            for mass in MASSES
            for rest in RESTS
            for precision in VALUES
        ),
        *(
            Projection(rank, precision, basis)
            for rank in RANKS
            for precision in COEFFICIENTS
        ),
    ]
