from pathlib import Path

import numpy as np
import pytest

from app.text_search import TextQuery, gemma_space, text_search


def test_text_search_ranks_the_row_nearest_the_query_first(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = gemma_space()
    monkeypatch.setattr(
        "app.text_search.embed_queries", lambda _: 2 * space.vectors[3:4]
    )

    galaxies, modalities, scores = text_search(TextQuery(text="a galaxy", matches=5))

    assert len(galaxies) == 5
    assert (galaxies[0], modalities[0]) == (space.galaxy[3], space.modality[3])
    assert np.all(np.diff(scores) <= 0)


def test_text_search_returns_a_galaxy_once_per_matching_modality(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = gemma_space()
    spectrum = np.flatnonzero(space.modality == "spectrum")[0]
    image = np.flatnonzero(
        (space.modality == "image") & (space.galaxy == space.galaxy[spectrum])
    )[0]
    monkeypatch.setattr(
        "app.text_search.embed_queries",
        lambda _: space.vectors[[image]] + space.vectors[[spectrum]],
    )

    galaxies, modalities, _ = text_search(TextQuery(text="a galaxy", matches=2))

    assert galaxies.tolist() == [space.galaxy[spectrum]] * 2
    assert sorted(modalities.tolist()) == ["image", "spectrum"]


def test_text_search_returns_every_row_when_asked_for_more(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = gemma_space()
    monkeypatch.setattr("app.text_search.embed_queries", lambda _: space.vectors[:1])

    found, _, _ = text_search(TextQuery(text="a galaxy", matches=128))

    assert sorted(found.tolist()) == sorted(space.galaxy.tolist())
