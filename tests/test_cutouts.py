"""The cutout artifact: its galaxy-ordered layout and what `cutout` returns."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from app.config import CROP_PIXELS, CUTOUTS, artifact, build_dir
from app.cutouts import cutout, cutouts, encode


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_cutout_returns_the_stored_bytes(tree: Path, galaxy: int) -> None:
    """Serving is a lookup, so row `g` must be what galaxy `g` gets back."""
    stored = pq.read_table(artifact("cutouts")).column("png")

    assert cutout(galaxy) == stored[galaxy].as_py()


def test_cutouts_are_cropped_to_the_configured_size(tree: Path) -> None:
    with Image.open(BytesIO(cutout(0))) as png:
        assert png.size == (CROP_PIXELS, CROP_PIXELS)
        assert png.format == "PNG"


def test_rows_out_of_galaxy_order_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row order is the lookup, so a shuffled artifact would serve wrong images."""
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    build_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array([1, 0], type=pa.int32()),
                "png": pa.array([b"second", b"first"]),
            },
            schema=CUTOUTS,
        ),
        artifact("cutouts"),
    )

    cutouts.cache_clear()
    try:
        with pytest.raises(ValueError, match="galaxy order"):
            cutouts()
    finally:
        cutouts.cache_clear()


def test_a_rectangular_source_is_cropped_about_its_centre() -> None:
    """The reason `encode` takes a box: a border derived from the width alone
    gives a non-square crop off the centre of a non-square source."""
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
