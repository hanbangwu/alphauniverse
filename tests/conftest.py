import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GALAXIES = 12


@pytest.fixture(scope="session")
def galaxies() -> int:
    return GALAXIES


@pytest.fixture(scope="session")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from scripts.fixture import build

    os.environ["ALPHAUNIVERSE_CACHE"] = str(tmp_path_factory.mktemp("artifacts"))
    return build(GALAXIES)


@pytest.fixture(scope="session")
def client(tree: Path) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as opened:
        yield opened
