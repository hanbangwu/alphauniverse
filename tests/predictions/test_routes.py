import io

import numpy as np
import pyarrow as pa
import pytest
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import LegacySurveyImage
from datasets import Dataset
from fastapi.testclient import TestClient
from PIL import Image

from app.config import (
    AION_SPECTRUM_GRID,
    DATASET_DESI_SUFFIX,
    DATASET_SDSS_SUFFIX,
    HSC_TABLE_COLUMNS,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    TOKEN_SURVEYS,
    artifact,
)
from app.dataset import dataset as dataset_module
from app.dataset.dataset import table_columns

ARROW = "application/vnd.apache.arrow.stream"


def _stored(role: str) -> pa.Table:
    return pa.ipc.open_file(artifact(role)).read_all()


def _aion_prediction(mode: str) -> np.ndarray:
    return np.load(artifact(f"aion_prediction_{mode}"))


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
