import numpy as np
import pytest
from astropy.table import Table
from fastapi.testclient import TestClient

from alphauniverse.config import (
    MODES,
    QUERY_PREFIX,
    WEMM_DIM,
)
from server.text_search.routes import query_embedder


@pytest.mark.usefixtures("identified")
def test_text_search_download_is_ecsv_with_each_match_and_its_query(
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
    assert response.headers["content-type"].startswith("text/plain")
    table = Table.read(response.text, format="ascii.ecsv")
    assert len(table) == 3
    assert table.meta["raw_query"] == "a galaxy"
    assert table.meta["augmented_query"] == QUERY_PREFIX.format("a galaxy")


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
