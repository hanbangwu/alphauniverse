from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import normalize

from .config import N_PATCHES, PROBE, GalaxyIndex

if TYPE_CHECKING:
    import faiss


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    galaxy: GalaxyIndex
    patches: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_PATCHES)], ...],
        Field(alias="p", min_length=1, max_length=N_PATCHES),
    ]
    matches: Annotated[int, Field(ge=1, le=128)] = 32


def search(
    query: Query, *, index: faiss.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = normalize(
        index.reconstruct_batch(
            query.galaxy * N_PATCHES + np.asarray(query.patches)
        ).mean(axis=0, keepdims=True)
    )

    ids = index.search(direction, PROBE)[1][0]
    found, first = np.unique(ids[ids >= 0] // N_PATCHES, return_index=True)
    ranked = found[np.argsort(first)]
    order = np.concatenate(
        ([query.galaxy], ranked[ranked != query.galaxy][: query.matches])
    ).astype(np.int32)

    maps = (
        index.reconstruct_batch(
            (order[:, None] * N_PATCHES + np.arange(N_PATCHES)).reshape(-1)
        )
        @ direction.T
    ).reshape(len(order), N_PATCHES)
    scores = maps.max(axis=1)
    by_score = np.concatenate(([0], 1 + np.argsort(-scores[1:], kind="stable")))
    return order[by_score], scores[by_score], maps[by_score]
