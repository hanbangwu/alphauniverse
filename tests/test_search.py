from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pytest

from app import search as search_module
from app.config import (
    ANCHOR,
    DIM,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    REDSHIFT_TABLE_VALUE,
)
from app.search import (
    N_HSC_TABLE_VALUES,
    N_LS_TABLE_VALUES,
    NLIST,
    Query,
    centroid,
    image_tokens,
    index,
    rank,
    search,
    source,
    starts,
    table_value_maps,
    with_hsc,
    with_redshift,
    with_spectrum,
)
from scripts.benchmarks.search_quality import exact_ranking
from scripts.fixture import build, forget

SCORE_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def built(tree) -> faiss.Index:
    return index()


def test_search_is_deterministic(built: faiss.Index) -> None:
    query = Query(galaxy=2, p=(7, 8))
    first = search(query, index=built)
    second = search(query, index=built)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_rank_keeps_the_query_first_and_each_row_together() -> None:
    order = np.array([7, 3, 9, 1, 5], dtype=np.int32)
    image_token_scores = np.array(
        [
            [0.5, 0.6, 0.1],
            [0.2, 0.1, 0.0],
            [-0.4, -0.2, -0.3],
            [0.4, 0.7, 0.2],
            [0.9, 0.5, 0.3],
        ],
        dtype=np.float32,
    )
    spectrum_token_scores = np.array(
        [[0.1, 0.2], [0.3, 0.25], [np.nan, np.nan], [0.8, 0.0], [0.1, 0.4]],
        dtype=np.float32,
    )
    rows = [0, 4, 3, 1, 2]
    expected = (
        order[rows],
        np.array([0.6, 0.9, 0.8, 0.3, -0.2], dtype=np.float32),
        image_token_scores[rows],
        spectrum_token_scores[rows],
    )

    for found, wanted in zip(
        rank(order, image_token_scores, spectrum_token_scores), expected, strict=True
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
    found, _, _, _, _ = search(
        Query(galaxy=0, p=(64, 65), matches=matches), index=built
    )

    assert found[0] == 0
    assert len(set(found.tolist())) == len(found) == min(matches, len(starts()) - 1) + 1


@pytest.mark.parametrize(
    "fields",
    [
        {"galaxy": 0, "p": (64, 65), "matches": 2},
        {"galaxy": 4, "p": (64, 65), "matches": 2},
        {"galaxy": 9, "s": (40, 41), "matches": 2},
        {"galaxy": 6, "p": (3,), "s": (100,), "matches": 2},
        {"galaxy": 0, "t": (2, 15), "matches": 2},
        {"galaxy": 6, "t": (REDSHIFT_TABLE_VALUE,), "matches": 2},
        {"galaxy": 0, "p": (64,), "t": (15, REDSHIFT_TABLE_VALUE), "matches": 2},
    ],
)
def test_approximate_ranking_agrees_with_exact(
    built: faiss.Index,
    fields: dict[str, int | tuple[int, ...]],
) -> None:
    query = Query.model_validate(fields)
    (
        expected,
        expected_scores,
        expected_maps,
        expected_spectral_maps,
        expected_table_value_maps,
    ) = exact_ranking(query.model_copy(update={"matches": query.matches + 1}))
    found, found_scores, maps, spectral_maps, found_table_value_maps = search(
        query, index=built
    )

    assert np.all(-np.diff(expected_scores[1:]) > 2 * SCORE_TOLERANCE)
    np.testing.assert_array_equal(found, expected[:-1])
    np.testing.assert_allclose(found_scores, expected_scores[:-1], atol=SCORE_TOLERANCE)
    np.testing.assert_allclose(maps, expected_maps[:-1], atol=SCORE_TOLERANCE)
    np.testing.assert_allclose(
        spectral_maps, expected_spectral_maps[:-1], atol=SCORE_TOLERANCE
    )
    np.testing.assert_allclose(
        found_table_value_maps, expected_table_value_maps[:-1], atol=SCORE_TOLERANCE
    )


def test_embeddings_that_are_not_finite_are_rejected() -> None:
    cells = pa.array([[[np.inf] * DIM]], type=pa.list_(pa.list_(pa.float16(), DIM)))

    with pytest.raises(ValueError, match="finite"):
        image_tokens(cells)


def test_ids_stay_contiguous_across_add_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(search_module, "BATCH", 1)

    galaxies = 9
    try:
        build(galaxies)

        built = index()
        assert (
            built.ntotal
            == galaxies * (N_IMAGE_TOKENS + N_LS_TABLE_VALUES)
            + with_spectrum().sum() * N_SPECTRUM_TOKENS
            + with_hsc().sum() * N_HSC_TABLE_VALUES
            + with_redshift().sum()
        )

        stored = built.reconstruct_batch(starts() + 7)
        rows = image_tokens(source("encoded").to_table(columns=["ls"]).column("ls"))
        expected = rows[np.arange(galaxies) * N_IMAGE_TOKENS + 7]

        np.testing.assert_allclose(stored, expected, atol=1e-3)
    finally:
        forget()


def test_selected_table_values_join_the_direction(built: faiss.Index) -> None:
    stored = source("encoded").take([0], columns=[ANCHOR, "hsc"])
    ls, hsc = (
        np.asarray(stored.column(survey)[0].as_py(), dtype=np.float32)
        for survey in (ANCHOR, "hsc")
    )
    table_values = np.stack((ls[N_IMAGE_TOKENS + 1], hsc[N_IMAGE_TOKENS + 2]))
    faiss.normalize_L2(table_values)
    expected = np.vstack(
        (built.reconstruct(int(starts()[0]) + 3)[None], table_values)
    ).mean(axis=0, keepdims=True)
    faiss.normalize_L2(expected)

    direction = centroid(Query(galaxy=0, p=(3,), t=(2, 15)), index=built)

    np.testing.assert_allclose(direction, expected, atol=SCORE_TOLERANCE)
    np.testing.assert_allclose(
        table_value_maps(np.array([0]), direction, index=built)[0, [2, 15]],
        (table_values @ direction.T)[:, 0],
        atol=SCORE_TOLERANCE,
    )
