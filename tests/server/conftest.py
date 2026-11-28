import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from datasets import Dataset
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def password() -> str:
    os.environ["ACCESS_PASSWORD"] = "test-password"
    return os.environ["ACCESS_PASSWORD"]


@pytest.fixture(scope="session")
def client(tree: Path, password: str) -> Iterator[TestClient]:
    from server.main import app

    headers = {"Authorization": f"Bearer {password}"}
    with TestClient(app, headers=headers) as opened:
        yield opened


@pytest.fixture
def identified(monkeypatch: pytest.MonkeyPatch, galaxies: int) -> Dataset:
    from alphauniverse.config import DATASET_LS_SUFFIX
    from scripts.fixture import identifiers
    from server.dataset import dataset as dataset_module

    rows = Dataset.from_dict(
        {
            f"object_id{DATASET_LS_SUFFIX}": identifiers(galaxies),
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
