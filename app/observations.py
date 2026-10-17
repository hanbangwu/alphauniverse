import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from astropy.stats import mad_std

from . import pql, templates
from .config import (
    BANDS,
    CROP_PIXELS,
    HSC,
    IMAGE_MODES,
    MODE_SURVEYS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    SPECTRUM_MODES,
    TABLE_VALUE_SURVEYS,
    TOKEN_SURVEYS,
)
from .dataset import dataset, redshift

MAGNITUDE_ERRORS = {
    f"{band}_cmodel_mag{HSC}": f"{band}_cmodel_magerr{HSC}" for band in "grizy"
}


def observed(mode: str) -> np.ndarray:
    return np.flatnonzero(pql.observed(MODE_SURVEYS[mode]))


def covering(column: pa.ChunkedArray, galaxies: np.ndarray) -> pa.Array:
    first = int(galaxies[0])
    return column.slice(first, int(galaxies[-1]) + 1 - first).combine_chunks()


def taken(column: pa.ChunkedArray, galaxies: np.ndarray) -> pa.Array:
    return covering(column, galaxies).take(pa.array(galaxies - galaxies[0]))


def column_values(column: str, galaxies: np.ndarray) -> np.ndarray:
    values = pc.fill_null(taken(dataset().data.column(column), galaxies), np.nan)
    return values.to_numpy().astype(np.float64)


def redshift_rows(galaxies: np.ndarray) -> list[dict]:
    columns = [column for survey in REDSHIFT_COLUMNS.values() for column in survey]
    return pa.table(
        {column: taken(dataset().data.column(column), galaxies) for column in columns}
    ).to_pylist()


def measured_redshifts(galaxies: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
    rows = redshift_rows(galaxies)
    surveys = [redshift(row) for row in rows]
    values = [
        np.nan if survey is None else row[REDSHIFT_COLUMNS[survey][0]]
        for row, survey in zip(rows, surveys)
    ]
    return np.array(values), surveys


def pixels(field: pa.ListArray, galaxies: int, bands: int) -> np.ndarray:
    values = field.flatten().flatten().flatten().to_numpy()
    size = round((len(values) / (galaxies * bands)) ** 0.5)
    start = round((size - CROP_PIXELS) / 2)
    crop = slice(start, start + CROP_PIXELS)
    return values.reshape(galaxies, bands, size, size)[:, :, crop, crop]


def images(survey: str, galaxies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = int(galaxies[0])
    cells = covering(dataset().data.column(TOKEN_SURVEYS[survey]), galaxies)
    found = np.flatnonzero(cells.is_valid().to_numpy(zero_copy_only=False))
    if not np.array_equal(found + first, galaxies):
        raise ValueError(f"{survey} images are not the galaxies observed in it")
    fields = dict(zip(cells.type.names, cells.flatten(), strict=True))
    names = np.asarray(fields["band"].flatten()).reshape(len(galaxies), -1)
    if not (np.char.upper(names.astype(str)) == BANDS[survey]).all():
        raise ValueError(f"{survey} bands are not stored as {BANDS[survey]}")
    bands = len(BANDS[survey])
    return (
        pixels(fields["flux"], len(galaxies), bands),
        pixels(fields["ivar"], len(galaxies), bands),
    )


def spectra(survey: str, galaxies: np.ndarray) -> pa.StructArray:
    return taken(dataset().data.column(TOKEN_SURVEYS[survey]), galaxies)


def samples(cells: pa.StructArray, row: int) -> tuple[np.ndarray, ...]:
    wavelength, flux, ivar, mask = (
        np.asarray(cells.field(field)[row].values)
        for field in ("lambda", "flux", "ivar", "mask")
    )
    valid = (wavelength > 0) & (ivar > 0) & ~mask
    return wavelength[valid], flux[valid], ivar[valid]


def gridded(
    cells: pa.StructArray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted = np.zeros((len(cells), len(grid)))
    information = np.zeros((len(cells), len(grid)))
    squares = np.zeros(len(cells))
    for row in range(len(cells)):
        wavelength, flux, ivar = samples(cells, row)
        position = np.interp(wavelength, grid, np.arange(len(grid)))
        low = np.minimum(position.astype(int), len(grid) - 2)
        upper = position - low
        for index, share in ((low, 1 - upper), (low + 1, upper)):
            weighted[row] += np.bincount(index, share * ivar * flux, len(grid))
            information[row] += np.bincount(index, share * ivar, len(grid))
        squares[row] = ivar @ flux**2
    return weighted, information, squares


def table_values(mode: str, galaxies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == REDSHIFT:
        rows = redshift_rows(galaxies)
        chosen = [REDSHIFT_COLUMNS[redshift(row)] for row in rows]
        return (
            np.array([[row[value]] for row, (value, _, _) in zip(rows, chosen)], float),
            np.array([[row[error]] for row, (_, error, _) in zip(rows, chosen)], float),
        )
    columns = TABLE_VALUE_SURVEYS[MODE_SURVEYS[mode]]
    values = np.stack([column_values(column, galaxies) for column in columns], axis=1)
    errors = [
        column_values(MAGNITUDE_ERRORS[column], galaxies)
        if column in MAGNITUDE_ERRORS
        else np.full(
            len(galaxies),
            mad_std(column_values(column, observed(mode)), ignore_nan=True),
        )
        for column in columns
    ]
    return values, np.stack(errors, axis=1)


def measured(
    mode: str, galaxies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode in SPECTRUM_MODES:
        grid = templates.decoder(mode).latent_grid.wavelength.cpu().numpy()
        return gridded(spectra(SPECTRUM_MODES[mode], galaxies), grid)
    if mode in IMAGE_MODES:
        flux, ivar = images(IMAGE_MODES[mode], galaxies)
        kept = ivar > 0
        information = np.where(kept, ivar, 0)
    else:
        flux, errors = table_values(mode, galaxies)
        kept = np.isfinite(flux) & (errors > 0)
        information = np.where(kept, errors, np.inf) ** -2.0
    values = np.where(kept, flux, 0).reshape(len(galaxies), -1)
    information = information.reshape(len(galaxies), -1)
    return information * values, information, (information * values**2).sum(axis=1)
