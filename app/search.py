from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.fs import LocalFileSystem

from .config import (
    ANCHOR,
    ARTIFACTS,
    N_IMAGE_TOKENS,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    TABLE_VALUE_SURVEYS,
    artifact,
)

N_LS_TABLE_VALUES = len(TABLE_VALUE_SURVEYS[ANCHOR])
N_HSC_TABLE_VALUES = len(TABLE_VALUE_SURVEYS["hsc"])
FIRST_LS_TABLE_VALUE = REDSHIFT_TABLE_VALUE + 1
FIRST_HSC_TABLE_VALUE = FIRST_LS_TABLE_VALUE + N_LS_TABLE_VALUES


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
def templates(mode: str) -> np.ndarray:
    return np.load(artifact(f"template_{mode}"), mmap_mode="r")


def table_value_tokens(galaxy: int) -> dict[str, int]:
    found = {
        column: int(token)
        for survey, columns in TABLE_VALUE_SURVEYS.items()
        if (cell := tokens().column(survey)[galaxy]).is_valid
        for column, token in zip(
            columns, np.asarray(cell.values)[N_IMAGE_TOKENS:], strict=True
        )
    }
    if (cell := tokens().column(REDSHIFT)[galaxy]).is_valid:
        found[REDSHIFT] = int(cell.values[0].as_py())
    return found


def table_value_templates(galaxy: int) -> dict[str, float]:
    found = {
        column: float(value)
        for survey, columns in TABLE_VALUE_SURVEYS.items()
        for column, value in zip(
            columns, templates(f"{survey}_table")[galaxy], strict=True
        )
    }
    found[REDSHIFT] = float(templates(REDSHIFT)[galaxy, 0])
    return found
