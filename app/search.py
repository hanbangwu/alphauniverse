from functools import cache
from typing import Annotated

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import normalize

from .config import (
    ANCHOR,
    BATCH,
    DIM,
    N_PATCHES,
    NLIST,
    NPROBE,
    PROBE,
    SEED,
    TRAIN_GALAXIES,
    GalaxyIndex,
    artifact,
)


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


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(artifact(role), format="parquet")


def patches(cells: pa.Array) -> np.ndarray:
    return normalize(
        np.asarray(
            pc.list_slice(cells, 0, N_PATCHES).flatten().flatten(), dtype=np.float32
        ).reshape(-1, DIM),
        copy=False,
    )


@cache
def index() -> faiss.Index:
    loaded = faiss.read_index(str(artifact("encoded_index")))
    loaded.make_direct_map()
    loaded.nprobe = NPROBE
    return loaded


def generate_index() -> None:
    dataset = source("encoded")
    sample = np.random.default_rng(SEED).choice(
        dataset.count_rows(), TRAIN_GALAXIES, replace=False
    )
    built = faiss.index_factory(DIM, f"IVF{NLIST},SQfp16", faiss.METRIC_INNER_PRODUCT)
    built.train(
        patches(dataset.take(sample, columns=[ANCHOR]).column(ANCHOR).combine_chunks())
    )
    for batch in dataset.to_batches(columns=[ANCHOR], batch_size=BATCH):
        built.add(patches(batch.column(ANCHOR)))

    faiss.write_index(built, str(artifact("encoded_index")))
