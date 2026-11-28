from functools import cache

import numpy as np
import pyarrow as pa

from alphauniverse.config import (
    AION_SPECTRUM_GRID,
    CATALOGS,
    DATASET_LS_SUFFIX,
    TOKEN_SURVEYS,
    Catalog,
    SpectrumSurvey,
)
from alphauniverse.dataset.dataset import dataset


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


def catalog(column: str) -> Catalog | None:
    _, separator, suffix = column.partition("-")
    return CATALOGS.get(separator + suffix)


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
        and catalog(field.name) is not None
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


def positions() -> pa.Table:
    return pa.table(
        {
            name: dataset().data.column(f"{name}{DATASET_LS_SUFFIX}")
            for name in ("object_id", "ra", "dec")
        }
    )
