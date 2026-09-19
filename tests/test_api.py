"""The HTTP contract: shapes, encodings, validation and error paths."""

import io

import numpy as np
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from app.config import (
    ARTIFACTS,
    DATASET_REVISION,
    GRID,
    N_MORPHOLOGIES,
    N_PATCHES,
    build_dir,
)
from scripts.fixture import TOKENS, covered


def test_meta_counts_every_galaxy(client: TestClient, galaxies: int) -> None:
    meta = client.get("/meta").json()

    assert meta["galaxies"] == galaxies
    assert meta["grid"] == GRID
    assert len(meta["morphologies"]) == N_MORPHOLOGIES
    assert sum(meta["morphologies"]) + meta["unlabelled"] == galaxies


def test_meta_names_the_artifact_roles_it_points_at(client: TestClient) -> None:
    meta = client.get("/meta").json()

    for role in ("embeddings", "mean_points", "full_points"):
        assert meta[role] in ARTIFACTS


def test_artifact_downloads(client: TestClient, tree) -> None:
    response = client.get("/artifacts/encoded")

    assert response.status_code == 200
    assert len(response.content) == (tree / "encoded.parquet").stat().st_size


def test_artifact_head_is_served(client: TestClient) -> None:
    assert client.head("/artifacts/mean_points").status_code == 200


def test_unknown_artifact_role_is_not_found(client: TestClient) -> None:
    assert client.get("/artifacts/nonsense").status_code == 404


def test_known_role_with_no_file_is_not_found(client: TestClient) -> None:
    """`codebook` is a real role the fixture does not build."""
    response = client.get("/artifacts/codebook")

    assert response.status_code == 404

    # This body goes to anyone who asks, and the `OSError` it used to carry
    # names the volume path and the revision.
    detail = response.json()["detail"]
    assert "codebook" in detail
    assert DATASET_REVISION not in detail
    assert str(build_dir()) not in detail


def test_tokens_are_one_uint32_per_patch(client: TestClient) -> None:
    response = client.get("/galaxies/0/tokens")

    assert response.status_code == 200
    values = np.frombuffer(response.content, dtype=np.uint32)
    assert values.shape == (N_PATCHES,)


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_coverage_reports_every_survey(client: TestClient, galaxy: int) -> None:
    rows = {
        row["survey"]: row["matched"]
        for row in client.get(f"/galaxies/{galaxy}/coverage").json()
    }

    for survey in TOKENS:
        assert rows[survey] is covered(survey, galaxy)
    assert {"gz10", "provabgs"} <= rows.keys()


def test_image_is_served_as_png(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cutout comes from the source dataset, which tests do not download."""
    from app import main

    monkeypatch.setattr(main, "image", lambda galaxy: b"\x89PNG-stub")
    response = client.get("/galaxies/0/image.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG-stub"


def _similarity(client: TestClient, **query) -> pa.RecordBatch:
    response = client.get("/similarity", params=query)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
    return pa.ipc.open_stream(io.BytesIO(response.content)).read_all()


def test_similarity_returns_one_arrow_batch(client: TestClient) -> None:
    table = _similarity(client, galaxy=2, p=[100, 101], matches=5)

    # Upper bound alone would pass on a response holding only the query galaxy.
    assert 1 < table.num_rows <= 6
    assert table.column_names == ["galaxy", "score", "map"]
    assert len(table.column("map")[0]) == N_PATCHES


def test_similarity_leads_with_the_query_galaxy(client: TestClient) -> None:
    table = _similarity(client, galaxy=7, p=[42])

    assert table.column("galaxy")[0].as_py() == 7
    scores = table.column("score").to_pylist()
    assert scores[1:] == sorted(scores[1:], reverse=True)


@pytest.mark.parametrize(
    "query",
    [
        {"galaxy": 0},
        {"galaxy": 0, "p": [N_PATCHES]},
        {"galaxy": 0, "p": [-1]},
        {"galaxy": -1, "p": [0]},
        {"galaxy": 0, "p": [0], "matches": 0},
        {"galaxy": 0, "p": [0], "matches": 129},
    ],
)
def test_similarity_rejects_invalid_queries(client: TestClient, query: dict) -> None:
    assert client.get("/similarity", params=query).status_code == 422


def test_negative_galaxy_is_rejected(client: TestClient) -> None:
    assert client.get("/galaxies/-1/tokens").status_code == 422


@pytest.mark.xfail(
    strict=True,
    reason="GalaxyIndex hardcodes the production galaxy count instead of "
    "deriving it from the artifacts, so rows past the end pass validation",
)
def test_galaxy_past_the_end_is_rejected(client: TestClient, galaxies: int) -> None:
    assert client.get(f"/galaxies/{galaxies + 1}/tokens").status_code == 422
