from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from app.config import CROP_PIXELS, artifact
from app.images import encode, image


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_image_returns_the_stored_bytes(tree: Path, galaxy: int) -> None:
    stored = pq.read_table(artifact("images")).column("png")

    assert image(galaxy) == stored[galaxy].as_py()


def test_images_are_cropped_to_the_configured_size(tree: Path) -> None:
    with Image.open(BytesIO(image(0))) as png:
        assert png.size == (CROP_PIXELS, CROP_PIXELS)
        assert png.format == "PNG"


def test_a_rectangular_source_is_cropped_about_its_centre() -> None:
    width, height = CROP_PIXELS + 104, CROP_PIXELS + 8
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - CROP_PIXELS) // 2, (height - CROP_PIXELS) // 2
    pixels[top, left] = (255, 0, 0)
    pixels[top + CROP_PIXELS - 1, left + CROP_PIXELS - 1] = (0, 0, 255)

    with Image.open(BytesIO(encode(Image.fromarray(pixels)))) as png:
        assert png.size == (CROP_PIXELS, CROP_PIXELS)
        assert png.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert png.convert("RGB").getpixel((CROP_PIXELS - 1, CROP_PIXELS - 1)) == (
            0,
            0,
            255,
        )
