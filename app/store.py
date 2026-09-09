from __future__ import annotations

from functools import cache

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.preprocessing import normalize

from .config import (
    ANCHOR,
    BATCH,
    DIM,
    N_PATCHES,
    NLIST,
    NPROBE,
    SEED,
    TRAIN_GALAXIES,
    artifact,
)


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(artifact(role), format="parquet")


def row(source: ds.Dataset, galaxy: int, columns: list[str]) -> pa.Table:
    return source.to_table(columns=columns, filter=ds.field("galaxy") == galaxy)


def vectors(cells: pa.ListArray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(cells.offsets, dtype=np.intp)
    return offsets - offsets[0], np.asarray(
        cells.flatten().flatten(), dtype=np.float32
    ).reshape(-1, DIM)


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
