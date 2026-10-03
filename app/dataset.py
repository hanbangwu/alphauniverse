from functools import cache
from io import BytesIO

import numpy as np
import PIL.Image
import pyarrow as pa
from datasets import Dataset, Image, load_dataset

from .config import (
    CROP_PIXELS,
    DATASET_AUTHOR,
    DATASET_NAME,
    DATASET_REVISION,
    FLAG_SURVEYS,
    N_MORPHOLOGIES,
    RGB_COLUMN,
    SPECTRUM_SURVEYS,
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


@cache
def labels() -> list[int]:
    category = dataset().data.column(FLAG_SURVEYS["gz10"])
    return np.bincount(
        np.asarray(category.fill_null(N_MORPHOLOGIES)), minlength=N_MORPHOLOGIES + 1
    ).tolist()
