import numpy as np
import torch

from app import pql
from app.config import TOP_CODES
from app.predictions import Mode, coefficients, image_token_codes

Stored = dict[str, torch.Tensor]


def kind(mode: Mode) -> str:
    return mode.name.split("_")[-1]


class Current:
    def __init__(
        self,
        mode: Mode,
        mean: torch.Tensor | None = None,
        directions: torch.Tensor | None = None,
    ) -> None:
        self.vocabulary = mode.vocabulary
        self.kind = kind(mode)
        self.mean = mean
        self.directions = directions
        self.floor = (
            pql.SPECTRUM_TOKEN_FLOOR
            if self.kind == "spectrum"
            else torch.finfo(torch.float32).tiny
        )

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        if self.kind not in ("image", "spectrum"):
            return {"values": log_probabilities.float().to(torch.float16)}
        shape = log_probabilities.shape[:-1]
        flat = log_probabilities.reshape(-1, log_probabilities.shape[-1])
        flat = flat.float().cpu().numpy()
        if self.kind == "image":
            parts = image_token_codes(flat)
            names = ("codes", "kept", "tails")
        else:
            parts = coefficients(
                np.exp(flat),
                self.mean.cpu().numpy(),
                self.directions.cpu().numpy(),
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

    def log_values(self, stored: Stored) -> torch.Tensor:
        return stored["values"].float()

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        if self.kind == "image":
            kept = pql.KEPT if side == "gallery" else TOP_CODES
            codes, probabilities, spread = self.image(stored, kept)
            dense = spread[..., None].expand(*spread.shape, self.vocabulary).clone()
            return dense.scatter_(-1, codes, probabilities)
        if self.kind == "spectrum":
            return self.mean + self.projected(stored) @ self.directions
        return self.log_values(stored).exp()

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        if self.kind == "image":
            parts = [part.cpu().numpy() for part in self.image(stored, pql.KEPT)]
            overlaps = pql.image_token_overlaps(query.cpu().numpy(), *parts)
            return torch.from_numpy(overlaps).to(query.device)
        if self.kind == "spectrum":
            return query @ self.mean + (
                self.projected(stored) * (query @ self.directions.T)
            ).sum(dim=-1)
        overlaps = pql.table_value_overlaps(
            self.log_values(stored).cpu().numpy(), query.log().cpu().numpy()
        )
        return torch.from_numpy(overlaps).exp().to(query.device)

    def bits(self, stored: Stored) -> int:
        return 8 * sum(
            array.numel() * array.element_size() for array in stored.values()
        )
