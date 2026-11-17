import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from aion.modalities import LegacySurveyImage
from datasets import Dataset
from fastapi.testclient import TestClient

from app.config import (
    DATASET_LS_SUFFIX,
    FLAG_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
)


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


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
