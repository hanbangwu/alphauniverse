import io

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

from app import dataset as dataset_module
from app import pql
from app.config import (
    ANCHOR,
    DESI,
    FLAG_SURVEYS,
    N_IMAGE_TOKENS,
    N_MORPHOLOGIES,
    N_SPECTRUM_TOKENS,
    N_TABLE_VALUES,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    SDSS,
    TABLE_VALUE_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
)
from app.dataset import table_columns

ARROW = "application/vnd.apache.arrow.stream"


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


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
    assert client.get("/downloads/predictions").status_code == 422


def test_known_role_with_no_file_is_not_found(client: TestClient) -> None:
    assert not artifact("codebook").exists()

    assert client.get("/downloads/codebook").status_code == 404


def test_image_tokens_are_that_galaxys_stored_image_token_ids(
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


def test_cosines_are_a_square_float32_table_of_the_mode_s_slots(
    client: TestClient,
) -> None:
    for mode, slots in (
        ("ls_image", N_IMAGE_TOKENS),
        ("desi_spectrum", N_SPECTRUM_TOKENS),
    ):
        response = client.get(f"/galaxy/0/cosines/{mode}")

        assert response.status_code == 200
        served = np.frombuffer(response.content, dtype=np.float32)
        np.testing.assert_allclose(
            served.reshape(slots, slots), pql.cosines(0, mode), err_msg=mode
        )


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


def test_search_returns_one_arrow_batch_with_every_mode(client: TestClient) -> None:
    table = _similarity(client, galaxy=2, ls_image=[100, 101], matches=5)

    assert 1 < table.num_rows <= 6
    assert table.column("galaxy")[0].as_py() == 2
    widths = {
        "ls_image": N_IMAGE_TOKENS,
        "hsc_image": N_IMAGE_TOKENS,
        "desi_spectrum": N_SPECTRUM_TOKENS,
        "sdss_spectrum": N_SPECTRUM_TOKENS,
        "table_values": N_TABLE_VALUES,
    }
    for name, width in widths.items():
        assert len(table.column(name)[0]) == width
        assert np.isfinite(np.asarray(table.column(name).to_pylist())).all()


def test_only_selected_modes_have_a_sum(client: TestClient) -> None:
    table = _similarity(
        client, galaxy=0, sdss_spectrum=[10, 11], table_values=[1], matches=5
    )

    sums = {
        name: table.column(name).is_valid().to_numpy().all()
        for name in table.column_names
        if name.endswith("_sum")
    }
    assert {name for name, valid in sums.items() if valid} == {
        "sdss_spectrum_sum",
        "ls_table_sum",
    }
    similar = {
        name
        for name in table.column_names
        if name.endswith("_similarity") and table.column(name).null_count == 0
    }
    assert similar == {"sdss_spectrum_similarity", "ls_table_similarity"}
    assert table.column("similarity").null_count == 0


def test_saliency_returns_one_row_with_every_map(client: TestClient) -> None:
    response = client.get("/saliency", params={"galaxy": 2, "ls_image": [100, 101]})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == ARROW
    table = pa.ipc.open_stream(io.BytesIO(response.content)).read_all()
    assert table.num_rows == 1
    assert {name: len(table.column(name)[0]) for name in table.column_names} == {
        "ls_image": N_IMAGE_TOKENS,
        "hsc_image": N_IMAGE_TOKENS,
        "desi_spectrum": N_SPECTRUM_TOKENS,
        "sdss_spectrum": N_SPECTRUM_TOKENS,
        "table_values": N_TABLE_VALUES,
    }


def test_only_position_independent_modes_have_positions(client: TestClient) -> None:
    table = _similarity(
        client,
        galaxy=0,
        ls_image=[3, 4],
        desi_spectrum=[10],
        anywhere=["ls_image"],
        matches=5,
    )

    assert table.column("ls_image_positions").null_count == 0
    assert {
        len(places) for places in table.column("ls_image_positions").to_pylist()
    } == {2}
    assert table.column("desi_spectrum_positions").null_count == table.num_rows


def test_has_flags_say_which_galaxies_observed_each_mode(client: TestClient) -> None:
    table = _similarity(client, galaxy=1, ls_image=[100], matches=11)
    galaxies = table.column("galaxy").to_numpy()
    stored = _stored("tokens")

    for column in ("hsc", "desi", "sdss", REDSHIFT):
        np.testing.assert_array_equal(
            table.column(f"has_{column}").to_numpy(zero_copy_only=False),
            stored.column(column).is_valid().to_numpy(zero_copy_only=False)[galaxies],
        )


@pytest.mark.parametrize(
    ("selection", "missing"),
    [
        ({"table_values": [13]}, "HSC match"),
        ({"hsc_image": [0]}, "HSC match"),
        ({"desi_spectrum": [0]}, "DESI spectrum"),
        ({"table_values": [REDSHIFT_TABLE_VALUE]}, "usable redshift"),
    ],
)
def test_a_selection_on_a_mode_the_galaxy_lacks_is_rejected(
    client: TestClient, selection: dict, missing: str
) -> None:
    response = client.get("/search", params={"galaxy": 1, **selection})

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert f"galaxy 1 has no {missing}" in error["msg"]


def test_the_redshift_is_searchable_for_a_galaxy_with_a_usable_redshift(
    client: TestClient,
) -> None:
    table = _similarity(client, galaxy=6, table_values=[REDSHIFT_TABLE_VALUE])

    assert table.column("redshift_sum").is_valid().to_numpy().all()


def test_table_rows_lead_with_the_redshifts_then_name_their_catalogue_and_table_value(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = Dataset.from_dict(
        {
            TABLE_VALUE_SURVEYS["hsc"][0]: [1.5],
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
    plain = {"table_value": None, "token": None, "excluded": None}
    assert response.status_code == 200
    assert response.json() == [
        {
            "section": REDSHIFT,
            "column": "DESI Z",
            "value": 0.25,
            "table_value": REDSHIFT_TABLE_VALUE,
            "token": redshift.values[0].as_py(),
            "excluded": None,
        },
        {"section": REDSHIFT, "column": "DESI ZERR", "value": 1e-4, **plain},
        {"section": REDSHIFT, "column": "DESI ZWARN", "value": False, **plain},
        {
            "section": REDSHIFT,
            "column": "SDSS Z",
            "value": 0.26,
            "table_value": None,
            "token": None,
            "excluded": "AION takes one redshift: DESI's",
        },
        {"section": REDSHIFT, "column": "SDSS Z_ERR", "value": 2e-4, **plain},
        {"section": REDSHIFT, "column": "SDSS ZWARNING", "value": False, **plain},
        {
            "section": "hsc",
            "column": "a_g",
            "value": 1.5,
            "table_value": 13,
            "token": hsc.values[N_IMAGE_TOKENS].as_py(),
            "excluded": None,
        },
    ]


def test_search_without_a_selection_is_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"galaxy": 0}).status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/galaxy/{galaxy}/image",
        "/galaxy/{galaxy}/image/tokens",
        "/galaxy/{galaxy}",
        "/galaxy/{galaxy}/spectrum",
        "/galaxy/{galaxy}/spectrum/tokens",
        "/galaxy/{galaxy}/table",
        "/search?galaxy={galaxy}&ls_image=0",
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
