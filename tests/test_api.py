import io

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

from app import dataset as dataset_module
from app.config import (
    ANCHOR,
    DESI,
    FLAG_SURVEYS,
    N_IMAGE_TOKENS,
    N_MORPHOLOGIES,
    REDSHIFT,
    REDSHIFT_SCALAR,
    SCALAR_SURVEYS,
    SDSS,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
)
from app.dataset import table_columns

ARROW = "application/vnd.apache.arrow.stream"


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


def _with_spectrum() -> np.ndarray:
    stored = _stored("encoded").select(list(SPECTRUM_SURVEYS))
    return np.logical_or.reduce([column.is_valid().to_numpy() for column in stored])


def test_artifact_downloads(client: TestClient) -> None:
    response = client.get("/downloads/encoded")

    assert response.status_code == 200
    assert len(response.content) == artifact("encoded").stat().st_size


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


def test_image_tokens_are_that_galaxys_stored_image_tokens(
    client: TestClient, galaxies: int
) -> None:
    galaxy = galaxies - 1
    cell = _stored("tokens").column(ANCHOR)[galaxy]

    response = client.get(f"/galaxy/{galaxy}/image/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[:N_IMAGE_TOKENS])


@pytest.mark.parametrize("galaxy", [0, 1, 6])
def test_coverage_reports_every_survey(client: TestClient, galaxy: int) -> None:
    stored = _stored("tokens").filter(pc.field("galaxy") == galaxy)
    rows = client.get(f"/galaxy/{galaxy}").json()

    assert rows.keys() == {*TOKEN_SURVEYS, *FLAG_SURVEYS}
    for survey in TOKEN_SURVEYS:
        assert rows[survey] is stored.column(survey)[0].is_valid


def test_unmatched_spectrum_tokens_are_not_found(client: TestClient) -> None:
    tokens = _stored("tokens").column("desi")
    untokenised = tokens.is_valid().to_pylist().index(False)

    assert client.get(f"/galaxy/{untokenised}/spectrum/tokens").status_code == 404


def test_spectrum_tokens_drop_the_normalisation_token(client: TestClient) -> None:
    column = _stored("tokens").column("desi")
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
    assert table.column_names == ["galaxy", "score", "map", "spectrum", "scalars"]
    assert len(table.column("map")[0]) == N_IMAGE_TOKENS


def test_similarity_spectrum_column_is_null_without_a_spectrum(
    client: TestClient,
) -> None:
    table = _similarity(client, galaxy=0, s=[10, 11], matches=11)
    galaxies = table.column("galaxy").to_numpy()

    np.testing.assert_array_equal(
        table.column("spectrum").is_valid().to_numpy(), _with_spectrum()[galaxies]
    )


def test_similarity_takes_scalars_alone(client: TestClient) -> None:
    table = _similarity(client, galaxy=0, t=[1, 13], matches=5)

    assert table.column("galaxy")[0].as_py() == 0
    assert 1 < table.num_rows <= 6


def test_every_galaxy_has_scalar_scores_where_it_has_scalars(
    client: TestClient,
) -> None:
    table = _similarity(client, galaxy=1, p=[100], matches=5)
    galaxies = table.column("galaxy").to_numpy()
    scores = np.asarray(table.column("scalars").to_pylist())
    stored = _stored("encoded")
    with_hsc = stored.column("hsc").is_valid().to_numpy()[galaxies]
    with_redshift = stored.column(REDSHIFT).is_valid().to_numpy()[galaxies]
    hsc = scores[:, 13:]

    assert np.isfinite(scores[:, 1:13]).all()
    np.testing.assert_array_equal(np.isfinite(hsc).all(axis=1), with_hsc)
    np.testing.assert_array_equal(np.isnan(hsc).all(axis=1), ~with_hsc)
    np.testing.assert_array_equal(
        np.isfinite(scores[:, REDSHIFT_SCALAR]), with_redshift
    )


def test_hsc_scalars_of_a_galaxy_without_hsc_are_rejected(
    client: TestClient,
) -> None:
    response = client.get("/search", params={"galaxy": 1, "t": [13]})

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert "galaxy 1 has no HSC match" in error["msg"]


def test_the_redshift_needs_a_usable_redshift_and_not_an_hsc_match(
    client: TestClient,
) -> None:
    accepted = client.get("/search", params={"galaxy": 6, "t": [REDSHIFT_SCALAR]})
    rejected = client.get("/search", params={"galaxy": 1, "t": [REDSHIFT_SCALAR]})

    assert accepted.status_code == 200
    assert rejected.status_code == 422
    [error] = rejected.json()["detail"]
    assert "galaxy 1 has no usable redshift" in error["msg"]


def test_table_rows_lead_with_the_redshifts_then_name_their_catalogue_and_scalar(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = Dataset.from_dict(
        {
            SCALAR_SURVEYS["hsc"][0]: [1.5],
            f"Z{DESI}": [0.25],
            f"ZERR{DESI}": [1e-4],
            f"ZWARN{DESI}": [False],
            f"Z{SDSS}": [0.26],
            f"Z_ERR{SDSS}": [2e-4],
            f"ZWARNING{SDSS}": [False],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    table_columns.cache_clear()

    response = client.get("/galaxy/0/table")
    table_columns.cache_clear()

    stored = _stored("tokens")
    hsc, redshift = (stored.column(name)[0] for name in ("hsc", REDSHIFT))
    plain = {"scalar": None, "token": None, "excluded": None}
    assert response.status_code == 200
    assert response.json() == [
        {
            "section": REDSHIFT,
            "column": "DESI Z",
            "value": 0.25,
            "scalar": REDSHIFT_SCALAR,
            "token": redshift.values[0].as_py(),
            "excluded": None,
        },
        {"section": REDSHIFT, "column": "DESI ZERR", "value": 1e-4, **plain},
        {"section": REDSHIFT, "column": "DESI ZWARN", "value": False, **plain},
        {
            "section": REDSHIFT,
            "column": "SDSS Z",
            "value": 0.26,
            "scalar": None,
            "token": None,
            "excluded": "AION takes one redshift: DESI's",
        },
        {"section": REDSHIFT, "column": "SDSS Z_ERR", "value": 2e-4, **plain},
        {"section": REDSHIFT, "column": "SDSS ZWARNING", "value": False, **plain},
        {
            "section": "hsc",
            "column": "a_g",
            "value": 1.5,
            "scalar": 13,
            "token": hsc.values[N_IMAGE_TOKENS].as_py(),
            "excluded": None,
        },
    ]


def test_similarity_without_image_tokens_or_spans_is_rejected(
    client: TestClient,
) -> None:
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
        "/galaxy/{galaxy}/table",
        "/search?galaxy={galaxy}&p=0",
    ],
)
def test_galaxy_past_the_end_is_rejected(
    client: TestClient, galaxies: int, path: str
) -> None:
    assert client.get(path.format(galaxy=galaxies)).status_code == 422


@pytest.mark.parametrize(
    "params", [{"text": ""}, {"text": "x" * 501}, {"text": "a", "matches": 0}]
)
def test_text_search_rejects_invalid_queries(client: TestClient, params: dict) -> None:
    assert client.get("/search/text", params=params).status_code == 422
