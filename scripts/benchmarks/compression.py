import numpy as np
import torch

from app import pql
from app.config import KEPT, PROBABILITY_FLOOR, TOP_CODES
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
            pql.SPECTRUM_TOKEN_FLOOR if self.kind == "spectrum" else PROBABILITY_FLOOR
        )

    def encode(self, log_probabilities: torch.Tensor) -> Stored:
        if self.kind not in ("image", "spectrum"):
            return {"values": log_probabilities.float().exp()}
        shape = log_probabilities.shape[:-1]
        flat = log_probabilities.reshape(-1, log_probabilities.shape[-1])
        flat = flat.float().cpu().numpy()
        if self.kind == "image":
            parts = image_token_codes(flat)
        else:
            parts = dict(
                zip(
                    ("quantised", "offsets", "steps"),
                    coefficients(
                        np.exp(flat),
                        self.mean.cpu().numpy(),
                        self.directions.cpu().numpy(),
                    ),
                    strict=True,
                )
            )
        return {
            name: torch.from_numpy(array)
            .reshape(*shape, *array.shape[1:])
            .to(log_probabilities.device)
            for name, array in parts.items()
        }

    def image(
        self, stored: Stored, side: str = "gallery"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if side == "gallery":
            return (
                stored["codes"].long(),
                stored["probabilities"],
                stored["rest"] / (self.vocabulary - KEPT),
            )
        return (
            torch.cat([stored["codes"].long(), stored["query_codes"].long()], dim=-1),
            torch.cat([stored["probabilities"], stored["query_probabilities"]], dim=-1),
            stored["tails"] / (self.vocabulary - TOP_CODES),
        )

    def projected(self, stored: Stored) -> torch.Tensor:
        return (
            stored["offsets"][..., None]
            + stored["quantised"] * stored["steps"][..., None]
        )

    def decode(self, stored: Stored, side: str = "gallery") -> torch.Tensor:
        if self.kind == "image":
            codes, probabilities, spread = self.image(stored, side)
            dense = spread[..., None].expand(*spread.shape, self.vocabulary).clone()
            return dense.scatter_(-1, codes, probabilities)
        if self.kind == "spectrum":
            return self.mean + self.projected(stored) @ self.directions
        return stored["values"]

    def overlap(self, stored: Stored, query: torch.Tensor) -> torch.Tensor:
        if self.kind == "image":
            parts = [part.cpu().numpy() for part in self.image(stored)]
            overlaps = pql.image_token_overlaps(query.cpu().numpy(), *parts)
            return torch.from_numpy(overlaps).to(query.device)
        if self.kind == "spectrum":
            return query @ self.mean + (
                self.projected(stored) * (query @ self.directions.T)
            ).sum(dim=-1)
        overlaps = pql.table_value_overlaps(
            stored["values"].cpu().numpy(), slice(None), query.cpu().numpy()
        )
        return torch.from_numpy(overlaps).to(query.device)

    def bits(self, stored: Stored) -> int:
        return 8 * sum(
            array.numel() * array.element_size() for array in stored.values()
        )
