from __future__ import annotations

from functools import cache
from io import BytesIO

from datasets import Dataset, Image, load_dataset
from PIL import ImageOps

from .config import CROP_PX, DATASET_ID, DATASET_REVISION, RGB_COLUMN


@cache
def dataset(dataset_id: str, dataset_revision: str) -> Dataset:
    return load_dataset(dataset_id, split="train", revision=dataset_revision)


def image(galaxy: int) -> bytes:
    data = dataset(DATASET_ID, DATASET_REVISION).select_columns([RGB_COLUMN])
    cutout = Image().decode_example(data[galaxy][RGB_COLUMN])
    crop = ImageOps.crop(cutout, (cutout.width - CROP_PX) // 2).convert("RGB")

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()
