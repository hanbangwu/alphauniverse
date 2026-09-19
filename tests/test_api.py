"""The HTTP contract: shapes, encodings, validation and error paths."""

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from app.config import ARTIFACTS, GRID, N_MORPHOLOGIES, N_PATCHES
from app.main import CACHE_CONTROL
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
    assert client.get("/artifacts/codebook").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/meta",
        "/artifacts/mean_points",
        "/galaxies/0/tokens",
        "/galaxies/0/coverage",
        "/galaxies/0/image.png",
        "/similarity?galaxy=0&p=0",
    ],
)
def test_responses_say_how_long_they_may_be_reused(
    client: TestClient, path: str
) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == CACHE_CONTROL


def test_head_responses_are_cacheable_too(client: TestClient) -> None:
    response = client.head("/artifacts/mean_points")

    assert response.status_code == 200
    assert response.headers["cache-control"] == CACHE_CONTROL


@pytest.mark.parametrize(
    ("call", "status"),
    [
        (lambda client: client.get("/artifacts/nonsense"), 404),
        (lambda client: client.post("/meta"), 405),
        (lambda client: client.get("/galaxies/99999999/tokens"), 422),
    ],
)
def test_errors_say_not_to_store_them(client: TestClient, call, status: int) -> None:
    """404 and 405 are both heuristically cacheable when no directive is set."""
    response = call(client)

    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"


def test_every_response_varies_on_origin(client: TestClient) -> None:
    """`public` without this lets a shared cache serve a copy carrying no CORS."""
    without = client.get("/meta")
    with_origin = client.get("/meta", headers={"Origin": "http://localhost:5173"})

    assert "origin" in without.headers["vary"].lower()
    assert with_origin.headers["vary"].lower().count("origin") == 1


def test_range_requests_still_work(client: TestClient, tree: Path) -> None:
    """DuckDB reads the point sets by range, so 206 must pass through intact."""
    response = client.get("/artifacts/mean_points", headers={"Range": "bytes=0-15"})

    assert response.status_code == 206
    assert response.content == (tree / "mean_points.parquet").read_bytes()[:16]
    assert response.headers["cache-control"] == CACHE_CONTROL


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


def test_image_is_served_as_png(client: TestClient) -> None:
    response = client.get("/galaxies/0/image.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


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
