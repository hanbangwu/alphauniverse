from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytest

from app import search as search_module
from app.config import (
    N_PATCHES,
    N_SPANS,
    NLIST,
    PROBE,
    SPECTRUM_SURVEYS,
    artifact,
)
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
from scripts.benchmark import observed_spans
from scripts.fixture import build
from scripts.recall import (
    Corpus,
    corpus,
    exact_ranking,
    match_counts,
    observed_mask,
    spectrum_coverage,
)

CACHES = (source, index, with_spectrum, starts)
SCORE_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


@pytest.fixture(scope="module")
def reference(tree) -> Corpus:
    return corpus()


def test_index_holds_every_patch_and_every_span(
    built: faiss.Index, galaxies: int
) -> None:
    stored = pq.read_table(artifact("encoded"), columns=list(SPECTRUM_SURVEYS))
    spectra = np.logical_or.reduce(
        [cell.is_valid().to_numpy() for cell in stored]
    ).sum()

    assert with_spectrum().sum() == spectra
    assert built.ntotal == galaxies * N_PATCHES + spectra * N_SPANS


@pytest.mark.parametrize("galaxy", [0, 5, 11])
@pytest.mark.parametrize("patch", [0, 1, N_PATCHES - 1])
def test_ids_encode_galaxy_and_patch(
    built: faiss.Index, galaxy: int, patch: int
) -> None:
    stored = built.reconstruct(int(starts()[galaxy]) + patch)
    cell = source("encoded").to_table(columns=["ls"]).column("ls")
    expected = patches(cell.combine_chunks())[galaxy * N_PATCHES + patch]

    np.testing.assert_allclose(stored, expected, atol=1e-3)


@pytest.mark.parametrize("galaxy", [0, 3, 8])
@pytest.mark.parametrize("span", [0, N_SPANS - 1])
def test_ids_encode_galaxy_and_span(built: faiss.Index, galaxy: int, span: int) -> None:
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


def test_rank_keeps_the_query_first_and_each_row_together() -> None:
    order = np.array([7, 3, 9, 1, 5], dtype=np.int32)
    patch_scores = np.array(
        [
            [0.5, 0.6, 0.1],
            [0.2, 0.1, 0.0],
            [-0.4, -0.2, -0.3],
            [0.4, 0.7, 0.2],
            [0.9, 0.5, 0.3],
        ],
        dtype=np.float32,
    )
    span_scores = np.array(
        [[0.1, 0.2], [0.3, 0.25], [np.nan, np.nan], [0.8, 0.0], [0.1, 0.4]],
        dtype=np.float32,
    )
    rows = [0, 4, 3, 1, 2]
    expected = (
        order[rows],
        np.array([0.6, 0.9, 0.8, 0.3, -0.2], dtype=np.float32),
        patch_scores[rows],
        span_scores[rows],
    )

    for found, wanted in zip(
        rank(order, patch_scores, span_scores), expected, strict=True
    ):
        np.testing.assert_array_equal(found, wanted, strict=True)


def test_asking_for_every_galaxy_returns_every_galaxy(
    built: faiss.Index, galaxies: int
) -> None:
    query = Query(galaxy=0, p=(64, 65), matches=galaxies - 1)
    found, _, _, _ = search(query, index=built)

    assert sorted(found.tolist()) == list(range(galaxies))


@pytest.mark.parametrize("matches", [3, 32])
@pytest.mark.parametrize(("nearest", "lists"), [(1, 1), (PROBE, 1), (1, NLIST)])
def test_a_search_that_finds_too_few_looks_further(
    built: faiss.Index,
    monkeypatch: pytest.MonkeyPatch,
    nearest: int,
    lists: int,
    matches: int,
) -> None:
    monkeypatch.setattr(search_module, "PROBE", nearest)
    monkeypatch.setattr(search_module, "NPROBE", lists)
    found, _, _, _ = search(Query(galaxy=0, p=(64, 65), matches=matches), index=built)

    assert len(set(found.tolist())) == len(found) == min(matches, len(starts()) - 1) + 1


@pytest.mark.parametrize(
    "fields",
    [
        {"galaxy": 0, "p": (64, 65), "matches": 2},
        {"galaxy": 4, "p": (64, 65), "matches": 2},
        {"galaxy": 9, "s": (40, 41), "matches": 2},
        {"galaxy": 6, "p": (3,), "s": (100,), "matches": 2},
    ],
)
def test_approximate_ranking_agrees_with_exact(
    built: faiss.Index,
    reference: Corpus,
    fields: dict[str, int | tuple[int, ...]],
) -> None:
    query = Query.model_validate(fields)
    expected, expected_scores, expected_maps, expected_spectral_maps = exact_ranking(
        query.model_copy(update={"matches": query.matches + 1}), reference
    )
    found, found_scores, maps, spectral_maps = search(query, index=built)

    assert np.all(-np.diff(expected_scores[1:]) > 2 * SCORE_TOLERANCE)
    np.testing.assert_array_equal(found, expected[:-1])
    np.testing.assert_allclose(found_scores, expected_scores[:-1], atol=SCORE_TOLERANCE)
    np.testing.assert_allclose(maps, expected_maps[:-1], atol=SCORE_TOLERANCE)
    np.testing.assert_allclose(
        spectral_maps, expected_spectral_maps[:-1], atol=SCORE_TOLERANCE
    )


def test_match_counts_count_spectra_and_best_tokens(built: faiss.Index) -> None:
    query = Query(galaxy=0, s=(1,), matches=11)
    galaxies, _, patch_maps, spectral_maps = (
        part[1:] for part in search(query, index=built)
    )
    tokens = pq.read_table(
        artifact("tokens"), columns=list(SPECTRUM_SURVEYS)
    ).to_pylist()
    held = [
        {survey for survey, cell in tokens[galaxy].items() if cell is not None}
        for galaxy in galaxies
    ]
    counts = {
        "desi": sum("desi" in surveys for surveys in held),
        "sdss_only": held.count({"sdss"}),
        "none": held.count(set()),
    }
    spanned = [row for row, surveys in enumerate(held) if "desi" in surveys]
    winners = [
        row for row in spanned if spectral_maps[row].max() > patch_maps[row].max()
    ]
    spectra = pq.read_table(artifact("spectra"), columns=["desi"]).column("desi")
    inside = sum(
        spectral_maps[row].argmax()
        in observed_spans(np.asarray(spectra[int(galaxies[row])]["wavelength"].values))
        for row in winners
    )

    assert all(counts.values())
    assert 0 < inside < len(winners)
    assert match_counts(
        spectrum_coverage(), observed_mask(), galaxies, patch_maps, spectral_maps
    ) == {
        "matches_with": counts,
        "best_is_span": len(winners),
        "best_span_observed": inside,
    }


def test_ids_stay_contiguous_across_add_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(search_module, "BATCH", 1)

    galaxies = 9
    for cache in CACHES:
        cache.cache_clear()

    try:
        build(galaxies)

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
