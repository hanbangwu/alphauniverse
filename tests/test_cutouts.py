"""The cutout artifact: its galaxy-ordered layout and what the endpoint returns."""

import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.config import CROP_PX, CUTOUTS, artifact, build_dir
from app.cutouts import cutouts, encode


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_endpoint_returns_the_stored_bytes(
    client: TestClient, tree: Path, galaxy: int
) -> None:
    """Serving is a lookup, so row `g` must be what galaxy `g` gets back."""
    stored = pq.read_table(artifact("cutouts")).column("png")

    response = client.get(f"/galaxies/{galaxy}/image.png")

    assert response.status_code == 200
    assert response.content == stored[galaxy].as_py()


def test_cutouts_are_cropped_to_the_configured_size(client: TestClient) -> None:
    with Image.open(BytesIO(client.get("/galaxies/0/image.png").content)) as png:
        assert png.size == (CROP_PX, CROP_PX)
        assert png.format == "PNG"


def test_each_galaxy_has_its_own_cutout(client: TestClient, galaxies: int) -> None:
    """Otherwise every assertion about which row is served passes vacuously."""
    served = {client.get(f"/galaxies/{g}/image.png").content for g in range(galaxies)}

    assert len(served) == galaxies


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
    width, height = CROP_PX + 104, CROP_PX + 8
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - CROP_PX) // 2, (height - CROP_PX) // 2
    pixels[top, left] = (255, 0, 0)
    pixels[top + CROP_PX - 1, left + CROP_PX - 1] = (0, 0, 255)

    with Image.open(BytesIO(encode(Image.fromarray(pixels)))) as png:
        assert png.size == (CROP_PX, CROP_PX)
        assert png.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert png.convert("RGB").getpixel((CROP_PX - 1, CROP_PX - 1)) == (0, 0, 255)


def test_a_short_artifact_stops_startup(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordered but short passes the order check, then `IndexError`s per request."""
    full = pq.read_table(artifact("cutouts"))

    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    build_dir().mkdir(parents=True, exist_ok=True)
    for role in ("mean_points", "tokens", "encoded", "encoded_index", "spectra"):
        shutil.copy(tree / artifact(role).name, artifact(role))
    pq.write_table(full.slice(0, full.num_rows - 1), artifact("cutouts"))

    cutouts.cache_clear()
    main.labels.cache_clear()
    try:
        with pytest.raises(ValueError, match="cutouts for"), TestClient(main.app):
            pass
    finally:
        cutouts.cache_clear()
        main.labels.cache_clear()
