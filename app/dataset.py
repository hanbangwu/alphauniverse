from functools import cache
from io import BytesIO

import numpy as np
import PIL.Image
import pyarrow as pa
from aion.fourm.modality_info import MODALITY_INFO
from astropy.convolution import Gaussian1DKernel, convolve
from datasets import Dataset, load_dataset, load_from_disk

from .config import (
    AION_SPECTRUM_GRID,
    CATALOGUES,
    DATASET_AUTHOR,
    DATASET_LS_SUFFIX,
    DATASET_NAME,
    DATASET_REVISION,
    HSC_BANDS,
    HSC_ZERO_POINT,
    LS_BANDS,
    LS_ZERO_POINT,
    REDSHIFT_COLUMNS,
    REDSHIFT_LIMIT,
    SPECTRUM_SMOOTHING_SIGMA,
    TOKEN_SURVEYS,
    Catalogue,
    ImageSurvey,
    SpectrumSurvey,
    dataset_dir,
)

RGB_SCALES = {"g": 6.0, "r": 3.4, "i": 3.0, "z": 2.2}
RGB_WEIGHTS = {
    "g": (0.0, 0.0, 0.75),
    "r": (0.0, 0.5, 0.25),
    "i": (0.25, 0.5, 0.0),
    "z": (0.75, 0.0, 0.0),
}
RGB_SOFTENING = 0.03
RGB_STRETCH = 20
HSC_TO_NANOMAGGIES = 10 ** (-0.4 * (HSC_ZERO_POINT - LS_ZERO_POINT))


@cache
def dataset() -> Dataset:
    if dataset_dir().exists():
        return load_from_disk(dataset_dir())
    return load_dataset(
        f"{DATASET_AUTHOR}/{DATASET_NAME}", split="train", revision=DATASET_REVISION
    )


def encode(image: PIL.Image.Image) -> bytes:
    left = (image.width - MODALITY_INFO["tok_image"]["input_size"]) // 2
    top = (image.height - MODALITY_INFO["tok_image"]["input_size"]) // 2
    crop = image.crop(
        (
            left,
            top,
            left + MODALITY_INFO["tok_image"]["input_size"],
            top + MODALITY_INFO["tok_image"]["input_size"],
        )
    ).convert("RGB")

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def colour(flux: np.ndarray, bands: list[str]) -> bytes:
    mapped = [band for band in bands if band in RGB_SCALES]
    scales = np.array([RGB_SCALES[band] for band in mapped])[:, None, None]
    scaled = flux[[bands.index(band) for band in mapped]] * scales + RGB_SOFTENING
    intensity = np.maximum(scaled, 0).mean(axis=0)
    stretch = np.divide(
        np.arcsinh(RGB_STRETCH * intensity) / np.sqrt(RGB_STRETCH),
        intensity,
        out=np.zeros_like(intensity),
        where=intensity > 0,
    )
    weights = np.array([RGB_WEIGHTS[band] for band in mapped])
    rgb = np.einsum("bhw,bc->hwc", np.clip(scaled * stretch, 0, 1), weights)
    return encode(PIL.Image.fromarray((rgb * 255).astype(np.uint8)))


def image(galaxy: int, survey: ImageSurvey) -> bytes | None:
    cell = dataset().data.column(TOKEN_SURVEYS[survey])[galaxy]
    if not cell.is_valid:
        return None
    bands = [band.partition("-")[2] for band in cell["band"].as_py()]
    rows = cell["flux"].values.flatten()
    flux = rows.flatten().to_numpy().reshape(len(bands), len(rows) // len(bands), -1)
    return colour(flux * (HSC_TO_NANOMAGGIES if survey == "hsc" else 1), bands)


def predicted_image(flux: np.ndarray, survey: ImageSurvey) -> bytes:
    bands = [
        band.partition("-")[2].lower()
        for band in (HSC_BANDS if survey == "hsc" else LS_BANDS)
    ]
    return colour(flux * (HSC_TO_NANOMAGGIES if survey == "hsc" else 1), bands)


def samples(cell: pa.StructScalar) -> dict[str, np.ndarray]:
    wavelength = np.asarray(cell["lambda"].values, dtype=np.float32)
    flux = np.asarray(cell["flux"].values, dtype=np.float32)
    kept = wavelength > 0
    flux = np.where(np.asarray(cell["mask"].values), np.nan, flux)
    return {"wavelength": wavelength[kept], "flux": flux[kept]}


def smoothed(flux: np.ndarray) -> np.ndarray:
    return convolve(flux, Gaussian1DKernel(SPECTRUM_SMOOTHING_SIGMA), preserve_nan=True)


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


def positions() -> pa.Table:
    return pa.table(
        {
            name: dataset().data.column(f"{name}{DATASET_LS_SUFFIX}")
            for name in ("ra", "dec")
        }
    )


def usable(redshift: float | None, warning: bool | None) -> bool:
    return redshift is not None and not warning and redshift <= REDSHIFT_LIMIT


def redshift(row: dict) -> SpectrumSurvey | None:
    return next(
        (
            survey
            for survey, (value, _, warning) in REDSHIFT_COLUMNS.items()
            if usable(row[value], row[warning])
        ),
        None,
    )
