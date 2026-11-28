import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from aion.modalities import DESISpectrum, LegacySurveyImage
from datasets import Dataset
from fastapi.testclient import TestClient

from alphauniverse.config import (
    DATASET_LS_SUFFIX,
    TOKEN_SURVEYS,
    artifact,
)


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


@pytest.mark.parametrize("galaxy", ["fixture-0", "fixture-1", "fixture-6"])
@pytest.mark.usefixtures("identified")
def test_coverage_reports_every_survey(client: TestClient, galaxy: str) -> None:
    stored = _stored("tokens").filter(pc.field("object_id") == galaxy)
    rows = client.get(f"/galaxy/{galaxy}").json()

    assert rows.keys() == {"object_id", "ra", "dec", *TOKEN_SURVEYS}
    for survey in TOKEN_SURVEYS:
        assert rows[survey] is stored.column(survey)[0].is_valid


def test_galaxy_is_named_by_its_legacy_survey_object_and_position(
    client: TestClient, identified: Dataset
) -> None:
    rows = client.get("/galaxy/fixture-3").json()

    assert (rows["object_id"], rows["ra"], rows["dec"]) == (
        identified[3][f"object_id{DATASET_LS_SUFFIX}"],
        identified[3][f"ra{DATASET_LS_SUFFIX}"],
        identified[3][f"dec{DATASET_LS_SUFFIX}"],
    )


@pytest.mark.parametrize(
    ("survey", "path"),
    [("hsc", "image/hsc/false_color"), ("desi", "spectrum/desi/false_color")],
)
def test_unmatched_observations_are_not_found(
    client: TestClient, survey: str, path: str
) -> None:
    tokens = _stored("tokens")
    untokenised = tokens.column(survey).is_valid().to_pylist().index(False)
    galaxy = tokens.column("object_id")[untokenised]

    assert client.get(f"/galaxy/{galaxy}/{path}").status_code == 404


@pytest.mark.parametrize(
    ("survey", "kind", "slots"),
    [
        ("ls", "image", LegacySurveyImage.num_tokens),
        ("hsc", "image", LegacySurveyImage.num_tokens),
        ("desi", "spectrum", DESISpectrum.num_tokens - 1),
        ("sdss", "spectrum", DESISpectrum.num_tokens - 1),
    ],
)
def test_false_color_is_an_rgb_byte_per_channel_per_slot(
    client: TestClient, survey: str, kind: str, slots: int
) -> None:
    tokens = _stored("tokens")
    galaxy = tokens.column("object_id")[
        int(np.flatnonzero(tokens.column(survey).is_valid())[-1])
    ]

    response = client.get(f"/galaxy/{galaxy}/{kind}/{survey}/false_color")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) == slots * 3


@pytest.mark.parametrize(
    "path",
    [
        "/galaxy/{galaxy}/image/ls",
        "/galaxy/{galaxy}/image/hsc/false_color",
        "/galaxy/{galaxy}/image/hsc/aion_prediction/false_color",
        "/galaxy/{galaxy}/image/hsc/aion_prediction",
        "/galaxy/{galaxy}",
        "/galaxy/{galaxy}/spectrum/sdss",
        "/galaxy/{galaxy}/spectrum/desi/false_color",
        "/galaxy/{galaxy}/spectrum/desi/aion_prediction/false_color",
        "/galaxy/{galaxy}/spectrum/sdss/aion_prediction",
        "/galaxy/{galaxy}/table",
    ],
)
def test_an_unknown_galaxy_is_not_found(client: TestClient, path: str) -> None:
    assert client.get(path.format(galaxy="unknown")).status_code == 404
