"""The cutout artifact: its galaxy-ordered layout and what the endpoint returns."""

from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import CROP_PX, CUTOUTS, artifact, build_dir
from app.cutouts import cutouts


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
