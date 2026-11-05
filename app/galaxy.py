from __future__ import annotations

from functools import cache
from typing import Annotated

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aion.modalities import LegacySurveyImage
from annotated_types import Predicate
from pydantic import Field

from core.artifacts import aion_predictions, artifact, tokens
from core.config import (
    AION_SPECTRUM_GRID,
    DATASET_LS_SUFFIX,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    REDSHIFT,
    TOKEN_SURVEYS,
    SpectrumSurvey,
)
from core.dataset import dataset, samples

from .config import CATALOGUES, REDSHIFT_TABLE_VALUE, Catalogue


@cache
def galaxy_count() -> int:
    return pq.read_metadata(artifact("mean_points")).num_rows


GalaxyIndex = Annotated[
    int, Field(ge=0), Predicate(lambda galaxy: galaxy < galaxy_count())
]


def spectrum(galaxy: int, survey: SpectrumSurvey) -> pa.Table | None:
    cell = dataset().data.column(TOKEN_SURVEYS[survey])[galaxy]
    if not cell.is_valid:
        return None
    return pa.table(samples(cell))


def fitted(shape: np.ndarray, galaxy: int, survey: SpectrumSurvey) -> np.ndarray:
    cell = dataset().data.column(TOKEN_SURVEYS[survey])[galaxy]
    if not cell.is_valid:
        return shape
    wavelength, flux, ivar, mask = (
        np.asarray(cell[field].values) for field in ("lambda", "flux", "ivar", "mask")
    )
    unmasked = (wavelength > 0) & ~mask
    model = np.interp(wavelength[unmasked], AION_SPECTRUM_GRID, shape)
    weights = ivar[unmasked]
    amplitude = (weights * flux[unmasked] * model).sum() / (weights * model**2).sum()
    return amplitude * shape


def catalogue(column: str) -> Catalogue | None:
    _, separator, suffix = column.partition("-")
    return CATALOGUES.get(separator + suffix)


@cache
def table_columns() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataset().data.schema
        if (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_boolean(field.type)
        )
        and catalogue(field.name) is not None
    )


def table(galaxy: int) -> dict[str, float | int | bool | None]:
    return {
        column: dataset().data.column(column)[galaxy].as_py()
        for column in table_columns()
    }


def identity(galaxy: int) -> dict[str, str | float]:
    return {
        name: dataset().data.column(f"{name}{DATASET_LS_SUFFIX}")[galaxy].as_py()
        for name in ("object_id", "ra", "dec")
    }


N_LS_TABLE_VALUES = len(LS_TABLE_COLUMNS)

N_HSC_TABLE_VALUES = len(HSC_TABLE_COLUMNS)

FIRST_LS_TABLE_VALUE = REDSHIFT_TABLE_VALUE + 1

FIRST_HSC_TABLE_VALUE = FIRST_LS_TABLE_VALUE + N_LS_TABLE_VALUES


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
