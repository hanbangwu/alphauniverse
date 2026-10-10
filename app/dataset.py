from functools import cache
from io import BytesIO

import numpy as np
import PIL.Image
import pyarrow as pa
from datasets import Dataset, Image, load_dataset

from .config import (
    CATALOGUES,
    CROP_PIXELS,
    DATASET_AUTHOR,
    DATASET_NAME,
    DATASET_REVISION,
    REDSHIFT_COLUMNS,
    REDSHIFT_RANGE,
    RGB_COLUMN,
    SPECTRUM_SURVEYS,
    Catalogue,
    SpectrumSurvey,
)


@cache
def dataset() -> Dataset:
    return load_dataset(
        f"{DATASET_AUTHOR}/{DATASET_NAME}", split="train", revision=DATASET_REVISION
    )


def encode(image: PIL.Image.Image) -> bytes:
    left = (image.width - CROP_PIXELS) // 2
    top = (image.height - CROP_PIXELS) // 2
    crop = image.crop((left, top, left + CROP_PIXELS, top + CROP_PIXELS)).convert("RGB")

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def image(galaxy: int) -> bytes:
    cell = dataset().data.column(RGB_COLUMN)[galaxy]
    return encode(Image().decode_example(cell.as_py()))


def samples(cell: pa.StructScalar) -> dict[str, np.ndarray]:
    wavelength = np.asarray(cell["lambda"].values, dtype=np.float32)
    flux = np.asarray(cell["flux"].values, dtype=np.float32)
    kept = wavelength > 0
    flux = np.where(np.asarray(cell["mask"].values), np.nan, flux)
    return {"wavelength": wavelength[kept], "flux": flux[kept]}


def spectrum(galaxy: int, survey: SpectrumSurvey) -> pa.Table | None:
    cell = dataset().data.column(SPECTRUM_SURVEYS[survey])[galaxy]
    if not cell.is_valid:
        return None
    return pa.table(samples(cell))


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


def usable(redshift: float | None, warning: bool | None) -> bool:
    return (
        redshift is not None
        and not warning
        and REDSHIFT_RANGE[0] <= redshift <= REDSHIFT_RANGE[1]
    )


def redshift(row: dict) -> SpectrumSurvey | None:
    return next(
        (
            survey
            for survey, (value, warning) in REDSHIFT_COLUMNS.items()
            if usable(row[value], row[warning])
        ),
        None,
    )
