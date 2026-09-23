"""The app's startup: label counts and the check that stops a short tree serving."""

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import artifact, build_dir
from app.cutouts import cutouts


def test_labels_partition_every_galaxy(tree: Path) -> None:
    galaxies, morphologies, unlabelled = main.labels()

    assert galaxies == pq.read_metadata(artifact("mean_points")).num_rows
    assert sum(morphologies) + unlabelled == galaxies


def test_a_short_artifact_stops_startup(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordered but short passes the order check, then `IndexError`s per request."""
    full = pq.read_table(artifact("cutouts"))

    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    build_dir().mkdir(parents=True, exist_ok=True)
    for role in ("mean_points", "tokens", "encoded", "encoded_index", "spectra"):
        shutil.copy(tree / artifact(role).name, artifact(role))
    pq.write_table(full.slice(0, full.num_rows - 1), artifact("cutouts"))

    cutouts.cache_clear()
    main.labels.cache_clear()
    try:
        with pytest.raises(ValueError, match="cutouts for"), TestClient(main.app):
            pass
    finally:
        cutouts.cache_clear()
        main.labels.cache_clear()
