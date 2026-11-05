from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from aion.modalities import LegacySurveyImage
from pyarrow.fs import LocalFileSystem

from .config import (
    ARTIFACTS,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    artifact,
)

N_LS_TABLE_VALUES = len(LS_TABLE_COLUMNS)
N_HSC_TABLE_VALUES = len(HSC_TABLE_COLUMNS)
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
def aion_predictions(mode: str) -> np.ndarray:
    return np.load(artifact(f"aion_prediction_{mode}"), mmap_mode="r")


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


def table_value_aion_predictions(galaxy: int) -> dict[str, float]:
    found = {
        column: float(value)
        for survey, columns in (("ls", LS_TABLE_COLUMNS), ("hsc", HSC_TABLE_COLUMNS))
        for column, value in zip(
            columns, aion_predictions(f"{survey}_table")[galaxy], strict=True
        )
    }
    found[REDSHIFT] = float(aion_predictions(REDSHIFT)[galaxy, 0])
    return found
