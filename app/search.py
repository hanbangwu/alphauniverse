from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import N_PATCHES, GalaxyIndex


class SearchMethod(StrEnum):
    CODEBOOK = "codebook"
    CODEBOOK_AND_ENCODED = "codebook_and_encoded"
    ENCODED = "encoded"


class CombineMode(StrEnum):
    AND = "and"
    OR = "or"


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    galaxy: GalaxyIndex
    patches: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_PATCHES)], ...],
        Field(alias="p", min_length=1, max_length=N_PATCHES),
    ]
    method: SearchMethod = SearchMethod.ENCODED
    weight: Annotated[float, Field(ge=0, le=1)] = 0.5
    combine: CombineMode = CombineMode.AND
    neighbours: Annotated[int, Field(ge=1, le=64)] = 8
    matches: Annotated[int, Field(ge=1, le=128)] = 32


def search(
    query: Query, *, encoded: np.ndarray, codebook: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maps = {}
    for name, store in (("encoded", encoded), ("codebook", codebook)):
        rows = store[query.galaxy * N_PATCHES : (query.galaxy + 1) * N_PATCHES][
            list(query.patches)
        ]
        if query.combine == CombineMode.AND:
            rows = rows.mean(axis=0, keepdims=True)
        directions = (
            rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        ).T
        norms = np.maximum(np.linalg.norm(store, axis=1, keepdims=True), 1e-12)
        maps[name] = (store @ directions / norms).max(axis=1).reshape(-1, N_PATCHES)
    on_encoded, on_codebook = maps["encoded"], maps["codebook"]

    match query.method:
        case SearchMethod.ENCODED:
            combined = on_encoded
        case SearchMethod.CODEBOOK:
            combined = on_codebook
        case SearchMethod.CODEBOOK_AND_ENCODED:
            combined = query.weight * on_codebook + (1 - query.weight) * on_encoded

    nearest = np.partition(combined, -query.neighbours, axis=1)
    best = nearest[:, -query.neighbours :].mean(axis=1)

    ranked = np.argsort(-best, kind="stable")
    order = np.concatenate(
        ([query.galaxy], ranked[ranked != query.galaxy][: query.matches])
    )

    return (
        order.astype(np.int32),
        best[order],
        on_encoded[order],
        on_codebook[order],
    )
