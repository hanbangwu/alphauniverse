import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from app.config import (
    ARTIFACTS,
    FLAG_SURVEYS,
    GRID,
    N_MORPHOLOGIES,
    N_PATCHES,
    N_SPANS,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
)

ARROW = "application/vnd.apache.arrow.stream"
ENDPOINTS = [
    "/artifacts/mean_points",
    "/meta",
    "/galaxies/0/image.png",
    "/galaxies/0/tokens",
    "/galaxies/0/coverage",
    "/galaxies/0/spectra/desi",
    "/galaxies/0/spectra/sdss/tokens",
    "/similarity?galaxy=0&p=0",
]


def _with_spectrum() -> np.ndarray:
    stored = pq.read_table(artifact("encoded"), columns=list(SPECTRUM_SURVEYS))
    return np.logical_or.reduce([column.is_valid().to_numpy() for column in stored])


def test_meta_counts_every_galaxy(client: TestClient, galaxies: int) -> None:
    meta = client.get("/meta").json()

    assert meta["galaxies"] == galaxies
    assert meta["grid"] == GRID
    assert len(meta["morphologies"]) == N_MORPHOLOGIES


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


def test_artifact_ranges_are_served(client: TestClient, tree) -> None:
    response = client.get("/artifacts/mean_points", headers={"Range": "bytes=0-9"})

    assert response.status_code == 206
    assert response.content == (tree / "mean_points.parquet").read_bytes()[:10]


@pytest.mark.parametrize("url", ENDPOINTS)
def test_a_request_with_the_current_etag_is_not_modified(
    client: TestClient, url: str
) -> None:
    served = client.get(url)

    unchanged = client.get(url, headers={"If-None-Match": served.headers["etag"]})
    stale = client.get(url, headers={"If-None-Match": '"stale"'})

    assert (unchanged.status_code, unchanged.content) == (304, b"")
    assert (stale.status_code, stale.content) == (200, served.content)


def test_an_artifact_head_with_the_current_etag_is_not_modified(
    client: TestClient,
) -> None:
    etag = client.head("/artifacts/mean_points").headers["etag"]

    response = client.head("/artifacts/mean_points", headers={"If-None-Match": etag})

    assert (response.status_code, response.content) == (304, b"")


def test_a_weak_etag_in_a_list_is_not_modified(client: TestClient) -> None:
    etag = client.get("/meta").headers["etag"]

    response = client.get("/meta", headers={"If-None-Match": f'"stale", W/{etag}'})

    assert response.status_code == 304


@pytest.mark.parametrize("url", ENDPOINTS)
def test_successful_responses_must_be_revalidated_before_reuse(
    client: TestClient, url: str
) -> None:
    assert client.get(url).headers["cache-control"] == "no-cache"


def test_unknown_artifact_role_is_not_found(client: TestClient) -> None:
    assert client.get("/artifacts/nonsense").status_code == 404


def test_known_role_with_no_file_is_not_found(client: TestClient) -> None:
    assert not artifact("codebook").exists()

    assert client.get("/artifacts/codebook").status_code == 404


def test_tokens_are_one_uint32_per_patch(client: TestClient) -> None:
    response = client.get("/galaxies/0/tokens")

    assert response.status_code == 200
    values = np.frombuffer(response.content, dtype=np.uint32)
    assert values.shape == (N_PATCHES,)


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_coverage_reports_every_survey(client: TestClient, galaxy: int) -> None:
    stored = pq.read_table(artifact("tokens"), filters=[("galaxy", "==", galaxy)])
    rows = {
        row["survey"]: row["matched"]
        for row in client.get(f"/galaxies/{galaxy}/coverage").json()
    }

    assert rows.keys() == {*TOKEN_SURVEYS, *FLAG_SURVEYS}
    for survey in TOKEN_SURVEYS:
        assert rows[survey] is stored.column(survey)[0].is_valid


def test_image_is_served_as_png(client: TestClient) -> None:
    response = client.get("/galaxies/0/image.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_spectrum_is_served_as_arrow(client: TestClient) -> None:
    response = client.get("/galaxies/0/spectra/desi")

    assert response.status_code == 200
    assert response.headers["content-type"] == ARROW
    table = pa.ipc.open_stream(io.BytesIO(response.content)).read_all()
    assert table.column_names == ["wavelength", "flux"]


def test_unmatched_spectrum_is_not_found(client: TestClient) -> None:
    spectra = pq.read_table(artifact("spectra"), columns=["desi"]).column("desi")
    tokens = pq.read_table(artifact("tokens"), columns=["desi"]).column("desi")
    without = spectra.is_valid().to_pylist().index(False)
    untokenised = tokens.is_valid().to_pylist().index(False)

    assert client.get(f"/galaxies/{without}/spectra/desi").status_code == 404
    assert client.get(f"/galaxies/{untokenised}/spectra/desi/tokens").status_code == 404


def test_spectrum_tokens_drop_the_normalisation_token(client: TestClient) -> None:
    cell = pq.read_table(artifact("tokens"), columns=["sdss"]).column("sdss")[0]

    response = client.get("/galaxies/0/spectra/sdss/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[1:])


def _similarity(client: TestClient, **query) -> pa.RecordBatch:
    response = client.get("/similarity", params=query)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == ARROW
    return pa.ipc.open_stream(io.BytesIO(response.content)).read_all()


def test_similarity_returns_one_arrow_batch(client: TestClient) -> None:
    table = _similarity(client, galaxy=2, p=[100, 101], matches=5)

    assert 1 < table.num_rows <= 6
    assert table.column_names == ["galaxy", "score", "map", "spectrum"]
    assert len(table.column("map")[0]) == N_PATCHES


def test_similarity_spectrum_column_is_null_without_a_spectrum(
    client: TestClient,
) -> None:
    table = _similarity(client, galaxy=0, s=[10, 11], matches=11)
    galaxies = table.column("galaxy").to_numpy()

    np.testing.assert_array_equal(
        table.column("spectrum").is_valid().to_numpy(), _with_spectrum()[galaxies]
    )


@pytest.mark.parametrize(
    "query",
    [
        {"galaxy": 0},
        {"galaxy": 0, "p": [N_PATCHES]},
        {"galaxy": 0, "p": [-1]},
        {"galaxy": 0, "s": [N_SPANS]},
        {"galaxy": -1, "p": [0]},
        {"galaxy": 0, "p": [0], "matches": 0},
        {"galaxy": 0, "p": [0], "matches": 129},
    ],
)
def test_similarity_rejects_invalid_queries(client: TestClient, query: dict) -> None:
    assert client.get("/similarity", params=query).status_code == 422


def test_spans_of_a_galaxy_without_a_spectrum_are_rejected(
    client: TestClient,
) -> None:
    galaxy = int(np.flatnonzero(~_with_spectrum())[0])

    response = client.get("/similarity", params={"galaxy": galaxy, "s": [0]})

    assert response.status_code == 422


def test_negative_galaxy_is_rejected(client: TestClient) -> None:
    assert client.get("/galaxies/-1/tokens").status_code == 422


@pytest.mark.xfail(
    strict=True,
    reason="GalaxyIndex hardcodes the production galaxy count instead of "
    "deriving it from the artifacts, so rows past the end pass validation",
)
def test_galaxy_past_the_end_is_rejected(client: TestClient, galaxies: int) -> None:
    assert client.get(f"/galaxies/{galaxies + 1}/tokens").status_code == 422
