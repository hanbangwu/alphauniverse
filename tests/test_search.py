from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pytest

from app import search as search_module
from app.config import (
    DIM,
    N_PATCHES,
    N_SPANS,
)
from app.search import (
    NLIST,
    Query,
    index,
    patches,
    rank,
    search,
    source,
    starts,
    with_spectrum,
)
from scripts.benchmarks.search_quality import (
    Corpus,
    corpus,
    exact_ranking,
)
from scripts.fixture import build, forget

SCORE_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


@pytest.fixture(scope="module")
def reference(tree) -> Corpus:
    return corpus()


def test_search_is_deterministic(built: faiss.Index) -> None:
    query = Query(galaxy=2, p=(7, 8))
    first = search(query, index=built)
    second = search(query, index=built)

    for left, right in zip(first, second, strict=True):
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


@pytest.mark.parametrize("matches", [3, 32])
@pytest.mark.parametrize(("nearest", "lists"), [(1, 1), (NLIST, 1), (1, NLIST)])
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

    assert found[0] == 0
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


def test_embeddings_that_are_not_finite_are_rejected() -> None:
    cells = pa.array([[[np.inf] * DIM]], type=pa.list_(pa.list_(pa.float16(), DIM)))

    with pytest.raises(ValueError, match="finite"):
        patches(cells)


def test_ids_stay_contiguous_across_add_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(search_module, "BATCH", 1)

    galaxies = 9
    try:
        build(galaxies)

        built = index()
        assert built.ntotal == galaxies * N_PATCHES + with_spectrum().sum() * N_SPANS

        stored = built.reconstruct_batch(starts() + 7)
        rows = patches(source("encoded").to_table(columns=["ls"]).column("ls"))
        expected = rows[np.arange(galaxies) * N_PATCHES + 7]

        np.testing.assert_allclose(stored, expected, atol=1e-3)
    finally:
        forget()
