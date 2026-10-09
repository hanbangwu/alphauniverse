from pathlib import Path

import numpy as np
import pytest

from app.text_search import TextQuery, aion_gemma_space, text_search


def test_text_search_ranks_the_galaxy_nearest_the_query_first(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vectors = aion_gemma_space()
    monkeypatch.setattr("app.text_search.embed_queries", lambda _: 2 * vectors[3:4])

    galaxies, scores = text_search(TextQuery(text="a galaxy", matches=5))

    assert len(galaxies) == 5
    assert galaxies[0] == 3
    assert np.all(np.diff(scores) <= 0)


def test_text_search_returns_every_galaxy_when_asked_for_more(
    tree: Path, galaxies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.text_search.embed_queries", lambda _: aion_gemma_space()[:1]
    )

    found, _ = text_search(TextQuery(text="a galaxy", matches=128))

    assert sorted(found.tolist()) == list(range(galaxies))
