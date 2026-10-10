from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from app import pql
from app.config import (
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    OBSERVATIONS,
    REDSHIFT,
    SPECTRUM_MODES,
    SPECTRUM_TOKEN_RANK,
    TABLE_MODES,
    TOP_CODES,
    VOCABULARY,
)

GALAXY = 0
SELECTION = {
    "ls_image": (0, 17, 300, 575),
    "hsc_image": (5, 6, 7),
    "desi_spectrum": (40, 41, 42),
    "sdss_spectrum": (100,),
    "table_values": (0, 2, 6, 15),
}
SELECTED_SLOTS = {
    "ls_image": [0, 17, 300, 575],
    "hsc_image": [5, 6, 7],
    "desi_spectrum": [40, 41, 42],
    "sdss_spectrum": [100],
    REDSHIFT: [0],
    "ls_table": [1, 5],
    "hsc_table": [2],
}


def column(rows: pa.Table, name: str, *shape: int) -> np.ndarray:
    values = rows[name].combine_chunks().flatten().to_numpy()
    return values.astype(np.float64).reshape(rows.num_rows, *shape)


def dense_image_tokens(rows: pa.Table, survey: str, kept: int) -> np.ndarray:
    codes = column(rows, f"{survey}_codes", N_IMAGE_TOKENS, TOP_CODES).astype(np.int64)
    probabilities = np.exp(
        column(rows, f"{survey}_log_probabilities", N_IMAGE_TOKENS, TOP_CODES)
    )
    rest = np.exp(column(rows, f"{survey}_tails", N_IMAGE_TOKENS)) + probabilities[
        ..., kept:
    ].sum(axis=-1)
    dense = np.repeat(
        (rest / (IMAGE_VOCABULARY - kept))[..., None], IMAGE_VOCABULARY, axis=-1
    )
    np.put_along_axis(dense, codes[..., :kept], probabilities[..., :kept], axis=-1)
    return dense


def dense_spectrum_tokens(rows: pa.Table, survey: str) -> np.ndarray:
    mean, directions, _, _ = pql.basis()[survey]
    coefficients = column(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS, 1) + column(
        rows, f"{survey}_coefficients", N_SPECTRUM_TOKENS, SPECTRUM_TOKEN_RANK
    ) * column(rows, f"{survey}_steps", N_SPECTRUM_TOKENS, 1)
    return mean + coefficients @ directions


def dense(rows: pa.Table, mode: str, kept: int) -> np.ndarray:
    survey, _, kind = mode.partition("_")
    if kind == "image":
        return dense_image_tokens(rows, survey, kept)
    if kind == "spectrum":
        return dense_spectrum_tokens(rows, survey)
    name, count, vocabulary = TABLE_MODES[mode]
    return np.exp(column(rows, name, count, vocabulary))


def log_overlaps(mode: str, gallery: np.ndarray, query: np.ndarray) -> np.ndarray:
    overlaps = (gallery * query).sum(axis=-1)
    if mode.endswith("spectrum"):
        overlaps = np.maximum(overlaps, pql.SPECTRUM_TOKEN_FLOOR)
    return np.log(overlaps)


@pytest.fixture(scope="module")
def table(tree: Path) -> pa.Table:
    return pql.predictions()


@pytest.fixture(scope="module")
def query(tree: Path) -> pql.Query:
    return pql.Query(galaxy=GALAXY, **SELECTION)


def test_mode_sums_equal_a_brute_force_computation_on_dense_distributions(
    table: pa.Table, query: pql.Query
) -> None:
    own = table.slice(GALAXY, 1)

    sums = pql.parts(query)

    assert sums.keys() == SELECTED_SLOTS.keys()
    for mode, slots in SELECTED_SLOTS.items():
        expected = log_overlaps(
            mode,
            dense(table, mode, pql.KEPT)[:, slots],
            dense(own, mode, TOP_CODES)[0, slots],
        ).sum(axis=1)
        np.testing.assert_allclose(sums[mode], expected, rtol=1e-4, err_msg=mode)


def test_aligned_maps_overlap_each_slot_with_the_same_slot_of_the_query(
    table: pa.Table, query: pql.Query
) -> None:
    galaxies = np.asarray([3, 0])
    own = table.slice(GALAXY, 1)
    shown = table.take(galaxies)

    aligned, _ = pql.maps(query, galaxies)

    expected = {
        mode: log_overlaps(
            mode, dense(shown, mode, pql.KEPT), dense(own, mode, TOP_CODES)[0]
        )
        for mode in SELECTED_SLOTS
    }
    assert aligned.keys() == {*IMAGE_MODES, *SPECTRUM_MODES, "table_values"}
    for mode in (*IMAGE_MODES, *SPECTRUM_MODES):
        np.testing.assert_allclose(aligned[mode], expected[mode], rtol=1e-4)
    np.testing.assert_allclose(
        aligned["table_values"],
        np.hstack([expected[mode] for mode in TABLE_MODES]),
        rtol=1e-4,
    )


def test_selection_maps_overlap_every_slot_with_the_mean_of_the_selected_slots(
    table: pa.Table, query: pql.Query
) -> None:
    galaxies = np.asarray([3, 0])
    own = table.slice(GALAXY, 1)
    shown = table.take(galaxies)

    _, selected = pql.maps(query, galaxies)

    assert selected.keys() == {*IMAGE_MODES, *SPECTRUM_MODES}
    for mode in selected:
        mean = dense(own, mode, TOP_CODES)[0, SELECTED_SLOTS[mode]].mean(axis=0)
        np.testing.assert_allclose(
            selected[mode],
            log_overlaps(mode, dense(shown, mode, pql.KEPT), mean),
            rtol=1e-4,
            err_msg=mode,
        )


def test_unselected_image_and_spectrum_modes_have_no_selection_map(
    tree: Path,
) -> None:
    aligned, selected = pql.maps(
        pql.Query(galaxy=GALAXY, table_values=(1,)), np.asarray([3])
    )

    assert selected == {}
    assert all(np.isfinite(values).all() for values in aligned.values())


def test_results_open_with_the_query_galaxy_then_the_best_other_galaxies(
    query: pql.Query,
) -> None:
    matches = 4
    ranked = query.model_copy(update={"matches": matches})

    results = pql.search(ranked)

    scores = pql.scores(ranked)
    others = np.delete(np.arange(len(scores)), GALAXY)
    best = others[np.argsort(-scores[others], kind="stable")][:matches]
    assert results.galaxies.tolist() == [GALAXY, *best.tolist()]
    np.testing.assert_allclose(results.scores, scores[results.galaxies])
    for mode, values in pql.parts(ranked).items():
        np.testing.assert_allclose(results.sums[mode], values[results.galaxies])


@pytest.mark.parametrize(
    "selected",
    [{"hsc_image": (0,)}, {"sdss_spectrum": (0,)}, {"table_values": (0,)}],
)
def test_a_selection_on_a_mode_the_query_galaxy_lacks_is_rejected(
    tree: Path, selected: dict[str, tuple[int, ...]]
) -> None:
    with pytest.raises(ValidationError):
        pql.Query(galaxy=1, **selected)


def test_a_selection_across_modes_ranks_by_its_mean_standardised_mode_sum(
    query: pql.Query,
) -> None:
    sums = pql.parts(query)

    expected = np.mean(
        [(values - values.mean()) / values.std() for values in sums.values()],
        axis=0,
    )
    np.testing.assert_allclose(pql.scores(query), expected)


def test_table_value_overlaps_stay_finite_where_float32_probabilities_underflow() -> (
    None
):
    query = np.full((1, VOCABULARY), -120.0, dtype=np.float32)
    gallery = np.full((3, 1, VOCABULARY), -120.0, dtype=np.float32)
    query[0, 0] = 0
    gallery[:, 0, 1] = 0

    found = pql.table_value_overlaps(gallery, query)

    expected = np.log(np.exp(gallery.astype(np.float64) + query).sum(axis=-1))
    np.testing.assert_allclose(found, expected, rtol=1e-6)


def test_a_repeated_slot_counts_once(tree: Path) -> None:
    repeated = pql.parts(
        pql.Query(galaxy=GALAXY, ls_image=(7, 7, 9), table_values=(2, 2))
    )
    once = pql.parts(pql.Query(galaxy=GALAXY, ls_image=(7, 9), table_values=(2,)))

    for mode, values in once.items():
        np.testing.assert_array_equal(repeated[mode], values)


def test_similarity_ranks_galaxies_as_the_score_does(query: pql.Query) -> None:
    totals = pql.parts(query)

    similarity = pql.similarity(pql.fractions(query, totals))

    np.testing.assert_array_equal(
        np.argsort(-similarity, kind="stable"),
        np.argsort(-pql.combine(totals), kind="stable"),
    )


def test_one_mode_similarity_is_the_mean_fraction_of_the_best_overlap(
    table: pa.Table,
) -> None:
    single = pql.Query(galaxy=GALAXY, table_values=(2, 6))
    own = dense(table.slice(GALAXY, 1), "ls_table", TOP_CODES)[0, [1, 5]]

    similarity = pql.similarity(pql.fractions(single, pql.parts(single)))

    overlaps = log_overlaps(
        "ls_table", dense(table, "ls_table", pql.KEPT)[:, [1, 5]], own
    )
    expected = np.exp((overlaps - np.log(own.max(axis=-1))).mean(axis=1))
    np.testing.assert_allclose(similarity, expected, rtol=1e-4)


def test_similarity_of_image_and_table_modes_is_at_most_one(query: pql.Query) -> None:
    results = pql.search(query.model_copy(update={"matches": 11}))

    for mode, values in results.similarities.items():
        if mode not in SPECTRUM_MODES:
            assert (values <= 1 + 1e-6).all(), mode


def test_predicted_lists_each_missing_observation_of_the_selection_once(
    query: pql.Query,
) -> None:
    results = pql.search(query.model_copy(update={"matches": 11}))

    for galaxy, missing in zip(
        results.galaxies.tolist(), results.predicted, strict=True
    ):
        assert missing == [
            column for column in OBSERVATIONS if not pql.observed(column)[galaxy]
        ]
