from collections.abc import Iterator
from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from aion.modalities import LegacySurveyImage
from pyarrow.fs import LocalFileSystem

from app.config import (
    AION_EMBEDDING_DIM,
    ARTIFACTS,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    REDSHIFT,
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
def observed(column: str) -> np.ndarray:
    return pc.is_valid(tokens().column(column)).to_numpy()


def table_value_tokens(galaxy: int) -> dict[str, int]:
    found = {
        column: int(token)
        for survey, columns in (("ls", LS_TABLE_COLUMNS), ("hsc", HSC_TABLE_COLUMNS))
        if (cell := tokens().column(survey)[galaxy]).is_valid
        for column, token in zip(
            columns,
            np.asarray(cell.values)[LegacySurveyImage.num_tokens :],
            strict=True,
        )
    }
    if (cell := tokens().column(REDSHIFT)[galaxy]).is_valid:
        found[REDSHIFT] = int(cell.values[0].as_py())
    return found


def stream(
    surveys: tuple[str, ...] = STORE_COLUMNS,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for survey in surveys:
        for stored in source("encoded").to_batches(
            columns=["galaxy", survey], batch_size=CHUNK
        ):
            batch = stored.filter(stored.column(survey).is_valid())
            cells = batch.column(survey)
            offsets = np.asarray(cells.offsets, dtype=np.intp)
            yield (
                np.asarray(batch.column("galaxy")),
                offsets - offsets[0],
                np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
                    -1, AION_EMBEDDING_DIM
                ),
            )
