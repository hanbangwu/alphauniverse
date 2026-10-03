from functools import cache
from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL.Image import Image

from .config import CROP_PIXELS, IMAGES, RGB_COLUMN, artifact


def encode(image: Image) -> bytes:
    left = (image.width - CROP_PIXELS) // 2
    top = (image.height - CROP_PIXELS) // 2
    crop = image.crop((left, top, left + CROP_PIXELS, top + CROP_PIXELS)).convert("RGB")

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def write_images(pngs: list[bytes]) -> None:
    pq.write_table(
        pa.table(
            {
                "galaxy": np.arange(len(pngs)),
                "png": pngs,
            },
            schema=IMAGES,
        ),
        artifact("images"),
    )


def generate_images() -> None:
    from datasets import Image

    from .dataset import dataset

    rows = dataset().select_columns([RGB_COLUMN])
    write_images([encode(Image().decode_example(row[RGB_COLUMN])) for row in rows])


@cache
def images() -> pa.ChunkedArray:
    return pq.read_table(artifact("images"), columns=["png"]).column("png")


def image(galaxy: int) -> bytes:
    return images()[galaxy].as_py()
