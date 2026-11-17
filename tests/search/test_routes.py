import io

import numpy as np
import pyarrow as pa
import pytest
from aion.modalities import DESISpectrum, LegacySurveyImage
from astropy.table import Table
from fastapi.testclient import TestClient

from app.config import (
    N_TABLE_VALUES,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    artifact,
)
from app.search import pql

ARROW = "application/vnd.apache.arrow.stream"


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


def _similarity(client: TestClient, path: str = "/search", **query) -> pa.RecordBatch:
    response = client.get(path, params=query)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == ARROW
    return pa.ipc.open_stream(io.BytesIO(response.content)).read_all()


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
    assert "DESI spectrum" in table.column("predicted")[0].as_py()
    assert len(table.column("ls_image")[0]) == LegacySurveyImage.num_tokens


def test_whole_mode_search_takes_any_position_only_in_a_selected_mode(
    client: TestClient,
) -> None:
    response = client.get(
        "/search/modes",
        params={"galaxy": 1, "modes": ["redshift"], "anywhere": ["ls_image"]},
    )

    assert response.status_code == 422


@pytest.mark.usefixtures("identified")
@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/search/download", {"ls_image": [25]}),
        ("/search/modes/download", {"modes": ["redshift"]}),
    ],
)
def test_search_downloads_are_ecsv_with_the_galaxy_and_its_matches(
    client: TestClient, path: str, query: dict
) -> None:
    response = client.get(path, params={"galaxy": 2, "matches": 5, **query})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert len(Table.read(response.text, format="ascii.ecsv")) == 6


def test_the_redshift_is_searchable_for_a_galaxy_with_a_usable_redshift(
    client: TestClient,
) -> None:
    table = _similarity(client, galaxy=6, table_values=[REDSHIFT_TABLE_VALUE])

    assert table.column("redshift_sum").is_valid().to_numpy().all()


def test_search_without_a_selection_is_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"galaxy": 0}).status_code == 422
