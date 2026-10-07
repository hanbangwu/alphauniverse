import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from app.config import (
    ANCHOR,
    FLAG_SURVEYS,
    N_MORPHOLOGIES,
    N_PATCHES,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
)

ARROW = "application/vnd.apache.arrow.stream"


def _with_spectrum() -> np.ndarray:
    stored = pq.read_table(artifact("encoded"), columns=list(SPECTRUM_SURVEYS))
    return np.logical_or.reduce([column.is_valid().to_numpy() for column in stored])


def test_artifact_downloads(client: TestClient, tree) -> None:
    response = client.get("/downloads/encoded")

    assert response.status_code == 200
    assert len(response.content) == (tree / "encoded.parquet").stat().st_size


def test_meta_reports_a_count_per_morphology_without_the_dataset(
    client: TestClient,
) -> None:
    response = client.get("/meta")

    assert response.status_code == 200
    assert len(response.json()["morphologies"]) == N_MORPHOLOGIES + 1


def test_projection_head_is_served(client: TestClient) -> None:
    assert client.head("/projections/mean").status_code == 200


@pytest.mark.parametrize("url", ["/projections/mean", "/galaxy/0"])
def test_a_request_with_the_current_etag_is_not_modified(
    client: TestClient, url: str
) -> None:
    served = client.get(url)

    unchanged = client.get(url, headers={"If-None-Match": served.headers["etag"]})
    stale = client.get(url, headers={"If-None-Match": '"stale"'})

    assert (unchanged.status_code, unchanged.content) == (304, b"")
    assert (stale.status_code, stale.content) == (200, served.content)


def test_a_weak_etag_in_a_list_is_not_modified(client: TestClient) -> None:
    etag = client.get("/galaxy/0").headers["etag"]

    response = client.get("/galaxy/0", headers={"If-None-Match": f'"stale", W/{etag}'})

    assert response.status_code == 304


def test_successful_responses_must_be_revalidated_before_reuse(
    client: TestClient,
) -> None:
    assert client.get("/galaxy/0").headers["cache-control"] == "no-cache"


def test_internal_artifact_is_not_downloadable(client: TestClient) -> None:
    assert client.get("/downloads/search_index").status_code == 422


def test_known_role_with_no_file_is_not_found(client: TestClient) -> None:
    assert not artifact("codebook").exists()

    assert client.get("/downloads/codebook").status_code == 404


def test_image_tokens_are_that_galaxys_stored_patch_tokens(
    client: TestClient, galaxies: int
) -> None:
    galaxy = galaxies - 1
    cell = pq.read_table(artifact("tokens"), columns=[ANCHOR]).column(ANCHOR)[galaxy]

    response = client.get(f"/galaxy/{galaxy}/image/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[:N_PATCHES])


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_coverage_reports_every_survey(client: TestClient, galaxy: int) -> None:
    stored = pq.read_table(artifact("tokens"), filters=[("galaxy", "==", galaxy)])
    rows = client.get(f"/galaxy/{galaxy}").json()

    assert rows.keys() == {*TOKEN_SURVEYS, *FLAG_SURVEYS}
    for survey in TOKEN_SURVEYS:
        assert rows[survey] is stored.column(survey)[0].is_valid


def test_unmatched_spectrum_tokens_are_not_found(client: TestClient) -> None:
    tokens = pq.read_table(artifact("tokens"), columns=["desi"]).column("desi")
    untokenised = tokens.is_valid().to_pylist().index(False)

    assert client.get(f"/galaxy/{untokenised}/spectrum/tokens").status_code == 404


def test_spectrum_tokens_drop_the_normalisation_token(client: TestClient) -> None:
    column = pq.read_table(artifact("tokens"), columns=["desi"]).column("desi")
    galaxy = column.is_valid().to_pylist().index(True)
    cell = column[galaxy]

    response = client.get(f"/galaxy/{galaxy}/spectrum/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[1:])


def _similarity(client: TestClient, **query) -> pa.RecordBatch:
    response = client.get("/search", params=query)
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


def test_similarity_without_patches_or_spans_is_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"galaxy": 0}).status_code == 422


def test_spans_of_a_galaxy_without_a_spectrum_are_rejected(
    client: TestClient,
) -> None:
    galaxy = int(np.flatnonzero(~_with_spectrum())[0])

    response = client.get("/search", params={"galaxy": galaxy, "s": [0]})

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert f"galaxy {galaxy} has no spectrum" in error["msg"]


@pytest.mark.parametrize(
    "path",
    [
        "/galaxy/{galaxy}/image",
        "/galaxy/{galaxy}/image/tokens",
        "/galaxy/{galaxy}",
        "/galaxy/{galaxy}/spectrum",
        "/galaxy/{galaxy}/spectrum/tokens",
        "/search?galaxy={galaxy}&p=0",
    ],
)
def test_galaxy_past_the_end_is_rejected(
    client: TestClient, galaxies: int, path: str
) -> None:
    assert client.get(path.format(galaxy=galaxies)).status_code == 422


def test_text_search_ranks_the_galaxy_nearest_the_query_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import text_search

    vectors = text_search.aion_gemma_space()
    monkeypatch.setattr(text_search, "embed_queries", lambda _: 2 * vectors[3:4])

    matches = client.get(
        "/search/text", params={"text": "a galaxy", "matches": 5}
    ).json()

    assert len(matches["galaxies"]) == 5
    assert matches["galaxies"][0] == 3
    assert np.all(np.diff(matches["scores"]) <= 0)


def test_text_search_returns_every_galaxy_when_asked_for_more(
    client: TestClient, galaxies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import text_search

    monkeypatch.setattr(
        text_search, "embed_queries", lambda _: text_search.aion_gemma_space()[:1]
    )

    matches = client.get(
        "/search/text", params={"text": "a galaxy", "matches": 128}
    ).json()

    assert sorted(matches["galaxies"]) == list(range(galaxies))


@pytest.mark.parametrize(
    "params", [{"text": ""}, {"text": "x" * 501}, {"text": "a", "matches": 0}]
)
def test_text_search_rejects_invalid_queries(client: TestClient, params: dict) -> None:
    assert client.get("/search/text", params=params).status_code == 422
