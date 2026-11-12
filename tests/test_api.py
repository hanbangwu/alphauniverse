import io
from pathlib import Path
from typing import get_args

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage
from datasets import Dataset
from fastapi.testclient import TestClient
from PIL import Image

from app import dataset as dataset_module
from app import pql
from app.config import (
    AION_SPECTRUM_GRID,
    DATASET_DESI_SUFFIX,
    DATASET_LS_SUFFIX,
    DATASET_SDSS_SUFFIX,
    FLAG_SURVEYS,
    HSC_TABLE_COLUMNS,
    MODES,
    N_TABLE_VALUES,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    TOKEN_SURVEYS,
    WEMM_DIM,
    Projection,
    artifact,
)
from app.dataset import table_columns
from app.main import PASSWORD_COOKIE, query_embedder

ARROW = "application/vnd.apache.arrow.stream"


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


def _aion_prediction(mode: str) -> np.ndarray:
    return np.load(artifact(f"aion_prediction_{mode}"))


@pytest.fixture
def identified(monkeypatch: pytest.MonkeyPatch, galaxies: int) -> Dataset:
    rows = Dataset.from_dict(
        {
            f"object_id{DATASET_LS_SUFFIX}": [
                f"1501p020-{galaxy}" for galaxy in range(galaxies)
            ],
            f"ra{DATASET_LS_SUFFIX}": [
                150.0 + galaxy / 100 for galaxy in range(galaxies)
            ],
            f"dec{DATASET_LS_SUFFIX}": [
                2.0 - galaxy / 100 for galaxy in range(galaxies)
            ],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    return rows


def test_properties_are_served_whole(client: TestClient) -> None:
    response = client.get("/galaxies/property")

    assert response.status_code == 200
    assert len(response.content) == artifact("properties").stat().st_size


def test_meta_is_served_without_the_dataset(client: TestClient) -> None:
    assert client.get("/meta").status_code == 200


def test_a_request_without_the_password_is_rejected(client: TestClient) -> None:
    assert TestClient(client.app).get("/meta").status_code == 401


def test_a_wrong_bearer_password_is_rejected(client: TestClient) -> None:
    headers = {"Authorization": "Bearer wrong"}

    assert client.get("/meta", headers=headers).status_code == 401


def test_a_wrong_cookie_is_rejected(client: TestClient) -> None:
    cookies = {PASSWORD_COOKIE: "wrong"}
    visitor = TestClient(client.app, cookies=cookies)

    assert visitor.get("/meta").status_code == 401


def test_login_sets_a_cookie_that_authenticates(
    client: TestClient, password: str
) -> None:
    visitor = TestClient(client.app, base_url="https://testserver")
    headers = {"Authorization": f"Bearer {password}"}

    assert visitor.post("/login", headers=headers).status_code == 200
    assert visitor.get("/meta").status_code == 200


def test_login_with_a_wrong_password_sets_no_cookie(client: TestClient) -> None:
    visitor = TestClient(client.app, base_url="https://testserver")
    headers = {"Authorization": "Bearer wrong"}

    assert visitor.post("/login", headers=headers).status_code == 401
    assert not visitor.cookies


@pytest.mark.parametrize("projection", get_args(Projection))
def test_projection_head_is_served(client: TestClient, projection: str) -> None:
    assert client.head(f"/projections/{projection}").status_code == 200


@pytest.mark.parametrize("url", ["/projections/mean", "/galaxy/0"])
@pytest.mark.usefixtures("identified")
def test_a_request_with_the_current_etag_is_not_modified(
    client: TestClient, url: str
) -> None:
    served = client.get(url)

    unchanged = client.get(url, headers={"If-None-Match": served.headers["etag"]})
    stale = client.get(url, headers={"If-None-Match": '"stale"'})

    assert (unchanged.status_code, unchanged.content) == (304, b"")
    assert (stale.status_code, stale.content) == (200, served.content)


@pytest.mark.usefixtures("identified")
def test_a_weak_etag_in_a_list_is_not_modified(client: TestClient) -> None:
    etag = client.get("/galaxy/0").headers["etag"]

    response = client.get("/galaxy/0", headers={"If-None-Match": f'"stale", W/{etag}'})

    assert response.status_code == 304


@pytest.mark.usefixtures("identified")
def test_successful_responses_must_be_revalidated_before_reuse(
    client: TestClient,
) -> None:
    assert client.get("/galaxy/0").headers["cache-control"] == "no-cache"


def test_missing_properties_file_is_not_found(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("app.main.artifact", lambda role: tmp_path / role)

    assert client.get("/galaxies/property").status_code == 404


@pytest.mark.parametrize("survey", ["ls", "hsc"])
def test_image_tokens_are_that_galaxys_stored_image_token_ids(
    client: TestClient, survey: str
) -> None:
    column = _stored("tokens").column(survey)
    galaxy = int(np.flatnonzero(column.is_valid())[-1])
    cell = column[galaxy]

    response = client.get(f"/galaxy/{galaxy}/image/{survey}/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(
        served, np.asarray(cell.values)[: LegacySurveyImage.num_tokens]
    )


@pytest.mark.parametrize("survey", ["ls", "hsc"])
def test_image_aion_prediction_is_a_png_of_the_crop(
    client: TestClient, survey: str
) -> None:
    response = client.get(f"/galaxy/0/image/{survey}/aion_prediction")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as png:
        assert png.size == (
            MODALITY_INFO["tok_image"]["input_size"],
            MODALITY_INFO["tok_image"]["input_size"],
        )


@pytest.mark.usefixtures("identified")
def test_positions_are_in_galaxy_order(client: TestClient) -> None:
    response = client.get("/positions")
    positions = pa.ipc.open_stream(response.content).read_all()

    assert response.headers["content-type"] == ARROW
    assert positions.num_rows == client.get("/meta").json()["galaxy_count"]
    galaxy = client.get("/galaxy/6").json()
    assert positions.column("ra")[6].as_py() == galaxy["ra"]
    assert positions.column("dec")[6].as_py() == galaxy["dec"]


@pytest.mark.parametrize("galaxy", [0, 1, 6])
@pytest.mark.usefixtures("identified")
def test_coverage_reports_every_survey(client: TestClient, galaxy: int) -> None:
    stored = _stored("tokens").filter(pc.field("galaxy") == galaxy)
    rows = client.get(f"/galaxy/{galaxy}").json()

    assert rows.keys() == {"object_id", "ra", "dec", *TOKEN_SURVEYS, *FLAG_SURVEYS}
    for survey in TOKEN_SURVEYS:
        assert rows[survey] is stored.column(survey)[0].is_valid


def test_galaxy_is_named_by_its_legacy_survey_object_and_position(
    client: TestClient, identified: Dataset
) -> None:
    rows = client.get("/galaxy/3").json()

    assert (rows["object_id"], rows["ra"], rows["dec"]) == (
        identified[3][f"object_id{DATASET_LS_SUFFIX}"],
        identified[3][f"ra{DATASET_LS_SUFFIX}"],
        identified[3][f"dec{DATASET_LS_SUFFIX}"],
    )


def test_cosines_are_a_square_float32_table_of_the_mode_s_slots(
    client: TestClient,
) -> None:
    for mode, slots in (
        ("ls_image", LegacySurveyImage.num_tokens),
        ("desi_spectrum", DESISpectrum.num_tokens - 1),
    ):
        response = client.get(f"/galaxy/0/cosines/{mode}")

        assert response.status_code == 200
        served = np.frombuffer(response.content, dtype=np.float32)
        np.testing.assert_allclose(
            served.reshape(slots, slots), pql.cosines(0, mode), err_msg=mode
        )


@pytest.mark.parametrize(
    ("survey", "path"), [("hsc", "image/hsc"), ("desi", "spectrum/desi")]
)
def test_unmatched_tokens_are_not_found(
    client: TestClient, survey: str, path: str
) -> None:
    tokens = _stored("tokens").column(survey)
    untokenised = tokens.is_valid().to_pylist().index(False)

    assert client.get(f"/galaxy/{untokenised}/{path}/tokens").status_code == 404


@pytest.mark.parametrize("survey", ["desi", "sdss"])
def test_spectrum_tokens_drop_the_normalisation_token(
    client: TestClient, survey: str
) -> None:
    column = _stored("tokens").column(survey)
    galaxy = int(np.flatnonzero(column.is_valid())[-1])
    cell = column[galaxy]

    response = client.get(f"/galaxy/{galaxy}/spectrum/{survey}/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[1:])


@pytest.mark.parametrize("survey", ["desi", "sdss"])
def test_spectrum_aion_prediction_without_that_spectrum_is_the_shape_on_the_codec_grid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, survey: str
) -> None:
    unobserved = Dataset.from_dict({TOKEN_SURVEYS[survey]: [None, None]})
    monkeypatch.setattr(dataset_module, "dataset", lambda: unobserved)

    response = client.get(f"/galaxy/1/spectrum/{survey}/aion_prediction")

    assert response.status_code == 200
    assert response.headers["content-type"] == ARROW
    table = pa.ipc.open_stream(io.BytesIO(response.content)).read_all()
    np.testing.assert_array_equal(
        table.column("wavelength").to_numpy(), AION_SPECTRUM_GRID.astype(np.float32)
    )
    np.testing.assert_array_equal(
        table.column("flux").to_numpy(), _aion_prediction(f"{survey}_spectrum")[1]
    )


def _similarity(client: TestClient, path: str = "/search", **query) -> pa.RecordBatch:
    response = client.get(path, params=query)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == ARROW
    return pa.ipc.open_stream(io.BytesIO(response.content)).read_all()


def test_search_returns_one_arrow_batch_with_every_mode(client: TestClient) -> None:
    table = _similarity(client, galaxy=2, ls_image=[100, 101], matches=5)

    assert 1 < table.num_rows <= 6
    assert table.column("galaxy")[0].as_py() == 2
    widths = {
        "ls_image": LegacySurveyImage.num_tokens,
        "hsc_image": LegacySurveyImage.num_tokens,
        "desi_spectrum": DESISpectrum.num_tokens - 1,
        "sdss_spectrum": DESISpectrum.num_tokens - 1,
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
        "ls_image": LegacySurveyImage.num_tokens,
        "hsc_image": LegacySurveyImage.num_tokens,
        "desi_spectrum": DESISpectrum.num_tokens - 1,
        "sdss_spectrum": DESISpectrum.num_tokens - 1,
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

    for column in ("ls", "hsc", "desi", "sdss", REDSHIFT):
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


def test_whole_mode_search_on_a_mode_the_galaxy_lacks_returns_each_selected_contribution(
    client: TestClient,
) -> None:
    table = _similarity(
        client,
        "/search/modes",
        galaxy=1,
        modes=["ls_image", "desi_spectrum"],
        anywhere=["desi_spectrum"],
        matches=5,
    )

    assert table.num_rows == 6
    assert table.column("galaxy")[0].as_py() == 1
    contributed = {
        name
        for name in table.column_names
        if name.endswith("_contribution") and table.column(name).null_count == 0
    }
    assert contributed == {"ls_image_contribution", "desi_spectrum_contribution"}
    assert len(table.column("ls_image")[0]) == LegacySurveyImage.num_tokens


def test_whole_mode_search_takes_any_position_only_in_a_selected_mode(
    client: TestClient,
) -> None:
    response = client.get(
        "/search/modes",
        params={"galaxy": 1, "modes": ["redshift"], "anywhere": ["ls_image"]},
    )

    assert response.status_code == 422


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
            HSC_TABLE_COLUMNS[0]: [1.5],
            f"Z{DATASET_DESI_SUFFIX}": [0.25],
            f"ZERR{DATASET_DESI_SUFFIX}": [1e-4],
            f"ZWARN{DATASET_DESI_SUFFIX}": [False],
            f"Z{DATASET_SDSS_SUFFIX}": [0.26],
            f"Z_ERR{DATASET_SDSS_SUFFIX}": [2e-4],
            f"ZWARNING{DATASET_SDSS_SUFFIX}": [False],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    table_columns.cache_clear()

    response = client.get("/galaxy/0/table")
    table_columns.cache_clear()

    stored = _stored("tokens")
    hsc, redshift = (stored.column(name)[0] for name in ("hsc", REDSHIFT))
    plain = {
        "aion_prediction": None,
        "table_value": None,
        "token": None,
        "excluded": None,
    }
    assert response.status_code == 200
    assert response.json() == [
        {
            "section": REDSHIFT,
            "column": "DESI Z",
            "value": 0.25,
            "aion_prediction": float(_aion_prediction(REDSHIFT)[0, 0]),
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
            "aion_prediction": None,
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
            "aion_prediction": float(_aion_prediction("hsc_table")[0, 0]),
            "table_value": 13,
            "token": hsc.values[LegacySurveyImage.num_tokens].as_py(),
            "excluded": None,
        },
    ]


@pytest.mark.parametrize(
    ("sdss_flagged", "beside"), [(False, "SDSS Z"), (True, "DESI Z")]
)
def test_aion_s_redshift_shows_beside_the_one_it_reads_or_else_desi_z(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sdss_flagged: bool,
    beside: str,
) -> None:
    rows = Dataset.from_dict(
        {
            f"Z{DATASET_DESI_SUFFIX}": [0.25],
            f"ZERR{DATASET_DESI_SUFFIX}": [1e-4],
            f"ZWARN{DATASET_DESI_SUFFIX}": [True],
            f"Z{DATASET_SDSS_SUFFIX}": [0.26],
            f"Z_ERR{DATASET_SDSS_SUFFIX}": [2e-4],
            f"ZWARNING{DATASET_SDSS_SUFFIX}": [sdss_flagged],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    table_columns.cache_clear()

    response = client.get("/galaxy/0/table")
    table_columns.cache_clear()

    aion_predictions = {
        row["column"]: row["aion_prediction"] for row in response.json()
    }
    assert aion_predictions == {
        column: float(_aion_prediction(REDSHIFT)[0, 0]) if column == beside else None
        for column in aion_predictions
    }


def test_search_without_a_selection_is_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"galaxy": 0}).status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/galaxy/{galaxy}/image/ls",
        "/galaxy/{galaxy}/image/hsc/tokens",
        "/galaxy/{galaxy}/image/hsc/aion_prediction",
        "/galaxy/{galaxy}",
        "/galaxy/{galaxy}/spectrum/sdss",
        "/galaxy/{galaxy}/spectrum/desi/tokens",
        "/galaxy/{galaxy}/spectrum/sdss/aion_prediction",
        "/galaxy/{galaxy}/table",
        "/search?galaxy={galaxy}&ls_image=0",
    ],
)
def test_galaxy_past_the_end_is_rejected(
    client: TestClient, galaxies: int, path: str
) -> None:
    assert client.get(path.format(galaxy=galaxies)).status_code == 422


def test_text_search_returns_each_match_with_its_contribution_from_every_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(0).standard_normal((1, WEMM_DIM)).astype(np.float32)
    monkeypatch.setitem(
        client.app.dependency_overrides, query_embedder, lambda: lambda _: query
    )

    response = client.get("/search/text", params={"text": "a galaxy", "matches": 3})

    assert response.status_code == 200
    found = response.json()
    assert len(found["galaxies"]) == len(set(found["galaxies"])) == 3
    assert len(found["scores"]) == 3
    assert set(found["contributions"]) == set(MODES)
    assert all(len(values) == 3 for values in found["contributions"].values())
