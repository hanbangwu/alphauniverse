import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GALAXIES = 12


def _forget() -> None:
    from app import cutouts, main, search, spectra

    search.source.cache_clear()
    search.index.cache_clear()
    search.with_spectrum.cache_clear()
    search.starts.cache_clear()
    main.labels.cache_clear()
    cutouts.cutouts.cache_clear()
    spectra.spectra.cache_clear()


@pytest.fixture(scope="session")
def galaxies() -> int:
    return GALAXIES


@pytest.fixture(scope="session")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    from scripts.fixture import build

    os.environ["ALPHAUNIVERSE_CACHE"] = str(tmp_path_factory.mktemp("artifacts"))
    _forget()
    target = build(GALAXIES)
    _forget()
    yield target
    _forget()


@pytest.fixture(scope="session")
def client(tree: Path) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as opened:
        yield opened
