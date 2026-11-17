from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from app.config import (
    Projection,
)


@pytest.mark.parametrize("projection", get_args(Projection))
def test_projection_head_is_served(client: TestClient, projection: str) -> None:
    assert client.head(f"/projections/{projection}").status_code == 200


def test_missing_properties_file_is_not_found(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("app.projection.routes.artifact", lambda role: tmp_path / role)

    assert client.get("/galaxies/property").status_code == 404
