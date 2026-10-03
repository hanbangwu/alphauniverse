from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SPECTRA, SPECTRUM_SURVEYS, artifact


def samples(cell: pa.StructScalar) -> dict[str, np.ndarray]:
    wavelength = np.asarray(cell["lambda"].values, dtype=np.float32)
    flux = np.asarray(cell["flux"].values, dtype=np.float32)
    kept = wavelength > 0
    flux = np.where(np.asarray(cell["mask"].values), np.nan, flux)
    return {"wavelength": wavelength[kept], "flux": flux[kept]}


def write_spectra(cells: dict[str, list[dict[str, np.ndarray] | None]]) -> None:
    galaxies = len(next(iter(cells.values())))
    pq.write_table(
        pa.table(cells | {"galaxy": np.arange(galaxies)}, schema=SPECTRA),
        artifact("spectra"),
        compression="zstd",
    )


def generate_spectra() -> None:
    from .dataset import dataset

    table = dataset().data
    write_spectra(
        {
            survey: [
                samples(cell) if cell.is_valid else None
                for cell in table.column(column)
            ]
            for survey, column in SPECTRUM_SURVEYS.items()
        }
    )


@cache
def spectra() -> pa.Table:
    return pq.read_table(artifact("spectra"))


def spectrum(galaxy: int, survey: str) -> pa.Table | None:
    cell = spectra().column(survey)[galaxy]
    if not cell.is_valid:
        return None
    return pa.table(
        {"wavelength": cell["wavelength"].values, "flux": cell["flux"].values}
    )
