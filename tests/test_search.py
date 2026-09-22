"""The search path: index layout, ranking contract and approximation quality."""

from pathlib import Path

import faiss
import numpy as np
import pytest

from app import search as search_module
from app.config import N_PATCHES, N_SPANS, SPECTRUM_SURVEYS
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
    span_maps,
    spectral,
    spectrum_cells,
    starts,
    vectors,
    with_spectrum,
)
from scripts.benchmark import exact_ranking
from scripts.fixture import build, covered

CACHES = (source, index, with_spectrum, starts)


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


def test_index_holds_every_patch_and_every_span(
    built: faiss.Index, galaxies: int
) -> None:
    spectra = sum(any(covered(s, g) for s in SPECTRUM_SURVEYS) for g in range(galaxies))

    assert with_spectrum().sum() == spectra
    assert built.ntotal == galaxies * N_PATCHES + spectra * N_SPANS


@pytest.mark.parametrize("galaxy", [0, 5, 11])
@pytest.mark.parametrize("patch", [0, 1, N_PATCHES - 1])
def test_ids_encode_galaxy_and_patch(
    built: faiss.Index, galaxy: int, patch: int
) -> None:
    """The block layout that search() relies on: patches lead each galaxy."""
    stored = built.reconstruct(int(starts()[galaxy]) + patch)
    cell = source("encoded").to_table(columns=["ls"]).column("ls")
    expected = patches(cell.combine_chunks())[galaxy * N_PATCHES + patch]

    np.testing.assert_allclose(stored, expected, atol=1e-3)


@pytest.mark.parametrize("galaxy", [0, 3, 8])
@pytest.mark.parametrize("span", [0, N_SPANS - 1])
def test_ids_encode_galaxy_and_span(built: faiss.Index, galaxy: int, span: int) -> None:
    """Spans follow the patches of a galaxy that has a spectrum."""
    stored = built.reconstruct(int(starts()[galaxy]) + N_PATCHES + span)
    cells = spectrum_cells(source("encoded").to_table(columns=list(SPECTRUM_SURVEYS)))
    expected = spectral(cells[galaxy : galaxy + 1])[span]

    np.testing.assert_allclose(stored, expected, atol=1e-3)


def test_query_galaxy_leads_the_ranking(built: faiss.Index) -> None:
    galaxies, scores, maps, spectral_maps = search(Query(galaxy=3, s=(7,)), index=built)

    assert galaxies[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-2)
    assert maps.shape == (len(galaxies), N_PATCHES)
    assert spectral_maps.shape == (len(galaxies), N_SPANS)


def test_span_maps_are_null_without_a_spectrum(built: faiss.Index) -> None:
    galaxies, _, _, spectral_maps = search(Query(galaxy=0, s=(1, 2)), index=built)

    assert np.array_equal(np.isnan(spectral_maps[:, 0]), ~with_spectrum()[galaxies])


def test_matches_are_sorted_by_descending_score(built: faiss.Index) -> None:
    _, scores, _, _ = search(Query(galaxy=0, p=(10, 11, 12)), index=built)

    assert np.all(np.diff(scores[1:]) <= 0)


def test_scores_are_each_row_s_best_token(built: faiss.Index) -> None:
    _, scores, maps, spectral_maps = search(Query(galaxy=1, p=(200,)), index=built)

    best = np.maximum(
        maps.max(axis=1), np.nan_to_num(spectral_maps, nan=-1).max(axis=1)
    )
    np.testing.assert_allclose(scores, best, rtol=1e-6)


def test_result_never_exceeds_the_requested_size(built: faiss.Index) -> None:
    galaxies, _, _, _ = search(Query(galaxy=0, p=(5,), matches=3), index=built)

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
    staged = rank(
        order,
        score_maps(vectors(order, index=built), direction, width=N_PATCHES),
        span_maps(order, direction, index=built),
    )

    for left, right in zip(search(query, index=built), staged, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "query",
    [
        Query(galaxy=0, p=(64, 65)),
        Query(galaxy=4, p=(64, 65)),
        Query(galaxy=9, s=(40, 41)),
        Query(galaxy=6, p=(3,), s=(100,)),
    ],
)
def test_approximate_ranking_agrees_with_exact(
    built: faiss.Index, query: Query, galaxies: int
) -> None:
    """Recall against brute force over every token.

    The fixture is small enough that the ANN candidate pool covers it, so the
    two rankings should agree exactly. On production-sized data they will not;
    scripts/benchmark.py measures the gap that remains.
    """
    query = query.model_copy(update={"matches": galaxies - 1})
    found, _, _, _ = search(query, index=built)
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
    One galaxy per batch also puts spectrum-less batches on the add path.
    """
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(search_module, "BATCH", 1)

    galaxies = 9
    for cache in CACHES:
        cache.cache_clear()

    try:
        build(galaxies)
        for cache in CACHES:
            cache.cache_clear()

        built = index()
        assert built.ntotal == galaxies * N_PATCHES + with_spectrum().sum() * N_SPANS

        stored = built.reconstruct_batch(starts() + 7)
        rows = patches(
            source("encoded").to_table(columns=["ls"]).column("ls").combine_chunks()
        )
        expected = rows[np.arange(galaxies) * N_PATCHES + 7]

        np.testing.assert_allclose(stored, expected, atol=1e-3)
    finally:
        for cache in CACHES:
            cache.cache_clear()
