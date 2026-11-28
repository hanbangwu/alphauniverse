from collections.abc import Iterator
from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow.fs import LocalFileSystem

from alphauniverse.config import (
    AION_EMBEDDING_DIM,
    ARTIFACTS,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    REDSHIFT_TABLE_VALUE,
    STORE_COLUMNS,
    artifact,
)

N_LS_TABLE_VALUES = len(LS_TABLE_COLUMNS)
N_HSC_TABLE_VALUES = len(HSC_TABLE_COLUMNS)
FIRST_LS_TABLE_VALUE = REDSHIFT_TABLE_VALUE + 1
FIRST_HSC_TABLE_VALUE = FIRST_LS_TABLE_VALUE + N_LS_TABLE_VALUES
CHUNK = 128


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(
        artifact(role),
        format=ARTIFACTS[role],
        filesystem=LocalFileSystem(use_mmap=True),
    )


@cache
def tokens() -> pa.Table:
    return source("tokens").to_table()


@cache
def object_ids() -> pa.Array:
    return tokens().column("object_id").combine_chunks()


def rows(identifiers: pa.Array | list[str]) -> np.ndarray:
    found = pc.index_in(identifiers, value_set=object_ids())
    if found.null_count:
        raise ValueError("unknown object_id")
    return found.to_numpy()


@cache
def observed(column: str) -> np.ndarray:
    return pc.is_valid(tokens().column(column)).to_numpy()


def stream(
    surveys: tuple[str, ...] = STORE_COLUMNS,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for survey in surveys:
        start = 0
        for stored in source("encoded").to_batches(columns=[survey], batch_size=CHUNK):
            valid = stored.column(survey).is_valid()
            batch = stored.filter(valid)
            cells = batch.column(survey)
            offsets = np.asarray(cells.offsets, dtype=np.intp)
            yield (
                start + np.flatnonzero(valid.to_numpy(zero_copy_only=False)),
                offsets - offsets[0],
                np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
                    -1, AION_EMBEDDING_DIM
                ),
            )
            start += stored.num_rows
