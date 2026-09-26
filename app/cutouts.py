from functools import cache
from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL.Image import Image as Cutout

from .config import (
    CROP_PIXELS,
    CUTOUTS,
    DATASET_ID,
    DATASET_REVISION,
    RGB_COLUMN,
    artifact,
    build_dir,
)


def encode(cutout: Cutout) -> bytes:
    left = (cutout.width - CROP_PIXELS) // 2
    top = (cutout.height - CROP_PIXELS) // 2
    crop = cutout.crop((left, top, left + CROP_PIXELS, top + CROP_PIXELS)).convert(
        "RGB"
    )

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def write_cutouts(pngs: list[bytes]) -> None:
    build_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array(np.arange(len(pngs)), type=pa.int32()),
                "png": pa.array(pngs, type=pa.large_binary()),
            },
            schema=CUTOUTS,
        ),
        artifact("cutouts"),
    )


def generate_cutouts() -> None:
    from datasets import Image

    from .dataset import dataset

    rows = dataset(DATASET_ID, DATASET_REVISION).select_columns([RGB_COLUMN])
    write_cutouts([encode(Image().decode_example(row[RGB_COLUMN])) for row in rows])


@cache
def cutouts() -> pa.ChunkedArray:
    table = pq.read_table(artifact("cutouts"))
    galaxies = table.column("galaxy").to_numpy(zero_copy_only=False)
    if not np.array_equal(galaxies, np.arange(len(galaxies))):
        raise ValueError(f"{artifact('cutouts')} is not in galaxy order")
    return table.column("png")


def cutout(galaxy: int) -> bytes:
    return cutouts()[galaxy].as_py()
