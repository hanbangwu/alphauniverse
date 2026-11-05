from collections.abc import Collection
from functools import cache

import numpy as np
import torch

from .artifacts import artifact
from .config import MODES, SEARCH_RANK


def scores(
    queries: torch.Tensor, found: dict[str, torch.Tensor], rank: int
) -> torch.Tensor:
    return queries[:, :rank] @ found["galleries"][:, :rank].T + found["offsets"]


@cache
def stored(name: str) -> dict[str, torch.Tensor]:
    with np.load(artifact(f"token_search_{name}")) as found:
        return {
            part: torch.from_numpy(found[part])
            for part in ("queries", "galleries", "offsets")
        }


def contributions(totals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        mode: (values - values.mean(axis=-1, keepdims=True))
        / values.std(axis=-1, keepdims=True)
        / len(totals)
        for mode, values in totals.items()
    }


def whole_mode_totals(
    galaxies: np.ndarray, modes: Collection[str], anywhere: Collection[str]
) -> dict[str, np.ndarray]:
    totals = {}
    for mode in MODES:
        if mode in modes:
            found = stored(f"{mode}_anywhere" if mode in anywhere else mode)
            matched = scores(found["queries"][galaxies], found, SEARCH_RANK)
            totals[mode] = matched.double().numpy()
    return totals
