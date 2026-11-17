import numpy as np
import pytest
from astropy.table import Table
from fastapi.testclient import TestClient

from app.config import (
    MODES,
    WEMM_DIM,
)
from app.text_search.routes import query_embedder


@pytest.mark.usefixtures("identified")
def test_text_search_download_is_csv_with_each_match(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(0).standard_normal((1, WEMM_DIM)).astype(np.float32)
    monkeypatch.setitem(
        client.app.dependency_overrides, query_embedder, lambda: lambda _: query
    )

    response = client.get(
        "/search/text/download", params={"text": "a galaxy", "matches": 3}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert len(Table.read(response.text, format="ascii.csv")) == 3


def test_text_search_returns_each_match_with_its_contribution_from_every_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(0).standard_normal((1, WEMM_DIM)).astype(np.float32)
    monkeypatch.setitem(
        client.app.dependency_overrides, query_embedder, lambda: lambda _: query
    )

    response = client.get("/search/text", params={"text": "a galaxy", "matches": 3})

    assert response.status_code == 200
    found = response.json()
    assert len(found["galaxies"]) == len(set(found["galaxies"])) == 3
    assert len(found["scores"]) == 3
    assert set(found["contributions"]) == set(MODES)
    assert all(len(values) == 3 for values in found["contributions"].values())
