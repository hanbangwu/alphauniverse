"""Precomputed spectra, one struct column per spectrum survey."""

from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    DATASET_ID,
    DATASET_REVISION,
    SPECTRA,
    SPECTRUM,
    SPECTRUM_SURVEYS,
    artifact,
    build_dir,
)


def samples(cell: pa.StructScalar) -> dict[str, np.ndarray]:
    """One survey's spectrum as float32 `wavelength` and `flux`."""
    wavelength = np.asarray(cell["lambda"].values, dtype=np.float32)
    flux = np.asarray(cell["flux"].values, dtype=np.float32)
    kept = wavelength > 0
    flux = np.where(np.asarray(cell["mask"].values), np.nan, flux)
    return {"wavelength": wavelength[kept], "flux": flux[kept]}


def write_spectra(cells: dict[str, list[dict[str, np.ndarray] | None]]) -> None:
    """Write `cells` to the spectra artifact, taking their order as galaxy order."""
    build_dir().mkdir(parents=True, exist_ok=True)
    galaxies = len(next(iter(cells.values())))
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array(np.arange(galaxies), type=pa.int32()),
                **{
                    survey: pa.array(cells[survey], type=SPECTRUM)
                    for survey in SPECTRUM_SURVEYS
                },
            },
            schema=SPECTRA,
        ),
        artifact("spectra"),
        compression="zstd",
    )


def generate_spectra() -> None:
    """Build stage: extract every galaxy's spectra, in row order."""
    from .dataset import dataset

    table = dataset(DATASET_ID, DATASET_REVISION).data
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
    """Every spectrum, read into memory once per process."""
    table = pq.read_table(artifact("spectra"))
    galaxies = table.column("galaxy").to_numpy()
    if not np.array_equal(galaxies, np.arange(len(galaxies))):
        raise ValueError(f"{artifact('spectra')} is not in galaxy order")
    return table


def spectrum(galaxy: int, survey: str) -> pa.Table | None:
    """The galaxy's samples from `survey`, or None where it was not matched."""
    cell = spectra().column(survey)[galaxy]
    if not cell.is_valid:
        return None
    return pa.table(
        {"wavelength": cell["wavelength"].values, "flux": cell["flux"].values}
    )
