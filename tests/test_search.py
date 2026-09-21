"""The search path: index layout, ranking contract and approximation quality."""

from pathlib import Path

import faiss
import numpy as np
import pytest

from app import search as search_module
from app.config import N_PATCHES
from app.search import (
    Query,
    candidates,
    centroid,
    index,
    patches,
    rank,
    score_maps,
    search,
    source,
    vectors,
)
from scripts.benchmark import exact_ranking
from scripts.fixture import build


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


def test_index_holds_every_anchor_patch(built: faiss.Index, galaxies: int) -> None:
    assert built.ntotal == galaxies * N_PATCHES


@pytest.mark.parametrize("galaxy", [0, 5, 11])
@pytest.mark.parametrize("patch", [0, 1, N_PATCHES - 1])
def test_faiss_ids_encode_galaxy_and_patch(
    built: faiss.Index, galaxy: int, patch: int
) -> None:
    """The `galaxy * N_PATCHES + patch` id layout that search() relies on."""
    stored = built.reconstruct(galaxy * N_PATCHES + patch)
    cell = source("encoded").to_table(columns=["ls"]).column("ls")
    expected = patches(cell.combine_chunks())[galaxy * N_PATCHES + patch]

    np.testing.assert_allclose(stored, expected, atol=1e-3)


def test_query_galaxy_leads_the_ranking(built: faiss.Index) -> None:
    galaxies, scores, maps = search(Query(galaxy=3, p=(100,)), index=built)

    assert galaxies[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-2)
    assert maps.shape == (len(galaxies), N_PATCHES)


def test_matches_are_sorted_by_descending_score(built: faiss.Index) -> None:
    _, scores, _ = search(Query(galaxy=0, p=(10, 11, 12)), index=built)

    assert np.all(np.diff(scores[1:]) <= 0)


def test_scores_are_each_row_s_best_patch(built: faiss.Index) -> None:
    _, scores, maps = search(Query(galaxy=1, p=(200,)), index=built)

    np.testing.assert_allclose(scores, maps.max(axis=1), rtol=1e-6)


def test_result_never_exceeds_the_requested_size(built: faiss.Index) -> None:
    galaxies, _, _ = search(Query(galaxy=0, p=(5,), matches=3), index=built)

    assert len(galaxies) <= 4
    assert len(set(galaxies.tolist())) == len(galaxies)


def test_search_is_deterministic(built: faiss.Index) -> None:
    query = Query(galaxy=2, p=(7, 8))
    first = search(query, index=built)
    second = search(query, index=built)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_stages_compose_into_search(built: faiss.Index) -> None:
    """`search` is its stages in order, which is what the benchmark times."""
    query = Query(galaxy=4, p=(30, 31))
    direction = centroid(query, index=built)
    order = candidates(query, direction, index=built)
    staged = rank(order, score_maps(vectors(order, index=built), direction))

    for left, right in zip(search(query, index=built), staged, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("galaxy", [0, 4, 9])
def test_approximate_ranking_agrees_with_exact(
    built: faiss.Index, galaxy: int, galaxies: int
) -> None:
    """Recall against brute force over every patch.

    The fixture is small enough that the ANN candidate pool covers it, so the
    two rankings should agree exactly. On production-sized data they will not;
    scripts/benchmark.py measures the gap that remains.
    """
    query = Query(galaxy=galaxy, p=(64, 65), matches=galaxies - 1)
    found, _, _ = search(query, index=built)
    expected, _ = exact_ranking(query)

    assert len(found) == galaxies

    assert set(found[1:].tolist()) == set(expected[1 : len(found)].tolist())


def test_ids_stay_contiguous_across_add_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id layout holds across several `add()` calls in `generate_index`.

    The shared fixture is smaller than `BATCH`, so it is built by a single
    `add()` and cannot exercise this. Contiguity across batches is the half of
    the invariant that a reordered or dropped batch would break, and it would
    break silently: every galaxy id in every `/similarity` response would shift.
    """
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(search_module, "BATCH", 3)

    galaxies = 9
    for cache in (source, index):
        cache.cache_clear()

    try:
        build(galaxies)
        for cache in (source, index):
            cache.cache_clear()

        built = index()
        assert built.ntotal == galaxies * N_PATCHES

        ids = np.arange(galaxies) * N_PATCHES + 7
        stored = built.reconstruct_batch(ids)
        rows = patches(
            source("encoded").to_table(columns=["ls"]).column("ls").combine_chunks()
        )
        expected = rows[ids]

        np.testing.assert_allclose(stored, expected, atol=1e-3)
    finally:
        for cache in (source, index):
            cache.cache_clear()
