from pathlib import Path

import faiss
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from sklearn.preprocessing import normalize

from app import search as search_module
from app.config import ANCHOR, N_PATCHES, N_SPANS, SPECTRUM_SURVEYS, artifact
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
from scripts.fixture import build

CACHES = (source, index, with_spectrum, starts)


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


def exact_ranking(query: Query) -> tuple[np.ndarray, np.ndarray]:
    cells = source("encoded").to_table(columns=[ANCHOR, *SPECTRUM_SURVEYS])
    spectra = spectrum_cells(cells)
    owners = np.repeat(
        np.flatnonzero(pc.is_valid(spectra).to_numpy(zero_copy_only=False)), N_SPANS
    )
    patch_rows = patches(cells.column(ANCHOR).combine_chunks())
    span_rows = spectral(spectra.drop_null())
    chosen = []
    if query.patches:
        chosen.append(patch_rows[query.galaxy * N_PATCHES + np.asarray(query.patches)])
    if query.spans:
        chosen.append(span_rows[owners == query.galaxy][np.asarray(query.spans)])
    direction = normalize(np.concatenate(chosen).mean(axis=0, keepdims=True))
    scores = (patch_rows @ direction.T).reshape(-1, N_PATCHES).max(axis=1)
    np.maximum.at(scores, owners, (span_rows @ direction.T).reshape(-1))
    order = np.argsort(-scores, kind="stable")
    chosen = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return chosen, scores[chosen]


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
    query = query.model_copy(update={"matches": galaxies - 1})
    found, _, _, _ = search(query, index=built)
    expected, _ = exact_ranking(query)

    assert len(found) == galaxies

    assert set(found[1:].tolist()) == set(expected[1 : len(found)].tolist())


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
