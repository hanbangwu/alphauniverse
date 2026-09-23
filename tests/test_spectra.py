"""The spectra artifact: its layout and what `spectrum` returns."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.config import SPECTRA, artifact, build_dir
from app.spectra import samples, spectra, spectrum


def test_spectrum_returns_the_stored_samples(tree: Path) -> None:
    """Serving is a lookup, so row `g` must be what galaxy `g` gets back."""
    stored = pq.read_table(artifact("spectra")).column("desi")[0]

    table = spectrum(0, "desi")

    np.testing.assert_array_equal(
        table.column("wavelength").to_numpy(), np.asarray(stored["wavelength"].values)
    )
    np.testing.assert_array_equal(
        table.column("flux").to_numpy(), np.asarray(stored["flux"].values)
    )


def test_unmatched_survey_has_no_spectrum(tree: Path) -> None:
    stored = pq.read_table(artifact("spectra"), columns=["desi"]).column("desi")

    assert spectrum(stored.is_valid().to_pylist().index(False), "desi") is None


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
