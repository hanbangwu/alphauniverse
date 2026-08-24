from __future__ import annotations

from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .config import ANCHOR, DIM, artifact


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


@cache
def matrix(role: str) -> np.ndarray:
    cells = source(role).to_table(columns=[ANCHOR]).column(ANCHOR).combine_chunks()
    return np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(-1, DIM)
