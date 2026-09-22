"""The spectra artifact: its layout and what the two endpoints return."""

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from app.config import SPECTRA, artifact, build_dir
from app.spectra import samples, spectra
from scripts.fixture import covered


def test_endpoint_returns_the_stored_samples(client: TestClient) -> None:
    """Serving is a lookup, so row `g` must be what galaxy `g` gets back."""
    stored = pq.read_table(artifact("spectra")).column("desi")[0]

    response = client.get("/galaxies/0/spectra/desi")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
    table = pa.ipc.open_stream(io.BytesIO(response.content)).read_all()
    assert table.column_names == ["wavelength", "flux"]
    assert table.column("wavelength").equals(
        pa.chunked_array([stored["wavelength"].values])
    )
    np.testing.assert_array_equal(
        table.column("flux").to_numpy(), np.asarray(stored["flux"].values)
    )


def test_unmatched_survey_is_not_found(client: TestClient) -> None:
    assert not covered("desi", 1)

    assert client.get("/galaxies/1/spectra/desi").status_code == 404
    assert client.get("/galaxies/1/spectra/desi/tokens").status_code == 404


def test_tokens_drop_the_normalisation_token(client: TestClient) -> None:
    cell = pq.read_table(artifact("tokens"), columns=["sdss"]).column("sdss")[0]

    response = client.get("/galaxies/0/spectra/sdss/tokens")

    assert response.status_code == 200
    served = np.frombuffer(response.content, dtype=np.uint32)
    np.testing.assert_array_equal(served, np.asarray(cell.values)[1:])


def test_samples_drop_padding_and_blank_masked_flux() -> None:
    cell = pa.scalar(
        {
            "lambda": [-1.0, 4000.0, 4001.0],
            "flux": [9.0, 1.0, 2.0],
            "mask": [True, False, True],
        }
    )

    kept = samples(cell)

    np.testing.assert_array_equal(kept["wavelength"], [4000.0, 4001.0])
    np.testing.assert_array_equal(kept["flux"], [1.0, np.nan])
    assert kept["flux"].dtype == np.float32


def test_rows_out_of_galaxy_order_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    build_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array([1, 0], type=pa.int32()),
                "desi": [None, None],
                "sdss": [None, None],
            },
            schema=SPECTRA,
        ),
        artifact("spectra"),
    )

    spectra.cache_clear()
    try:
        with pytest.raises(ValueError, match="galaxy order"):
            spectra()
    finally:
        spectra.cache_clear()
