from pathlib import Path
from threading import Thread

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


def slot_similarities(mode: str, gallery: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.exp(log_overlaps(mode, gallery, query) - np.log(query.max(axis=-1)))


def mode_sums(query: pql.Query) -> dict[str, np.ndarray]:
    return pql.column_sums(pql.agreements(query.galaxy), pql.selection(query))


def similarity(query: pql.Query) -> np.ndarray:
    found = pql.agreements(query.galaxy)
    selected = pql.selection(query)
    return pql.similarity(
        pql.fractions(
            pql.selected_peaks(found, selected), pql.column_sums(found, selected)
        )
    )


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

    sums = mode_sums(query)

    assert sums.keys() == SELECTED_SLOTS.keys()
    for mode, slots in SELECTED_SLOTS.items():
        expected = log_overlaps(
            mode,
            dense(table, mode, pql.KEPT)[:, slots],
            dense(own, mode, TOP_CODES)[0, slots],
        ).sum(axis=1)
        np.testing.assert_allclose(sums[mode], expected, rtol=1e-4, err_msg=mode)


def test_aligned_maps_compare_each_slot_with_the_same_slot_of_the_query(
    table: pa.Table, query: pql.Query
) -> None:
    galaxies = np.asarray([3, 0])
    own = table.slice(GALAXY, 1)
    shown = table.take(galaxies)

    aligned = pql.maps(pql.agreements(GALAXY), galaxies)

    expected = {
        mode: slot_similarities(
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


def test_a_position_independent_mode_sums_each_selected_slot_s_best_overlap(
    table: pa.Table,
) -> None:
    own = table.slice(GALAXY, 1)
    selected = {"ls_image": (0, 17, 300), "desi_spectrum": (40, 41, 42)}
    query = pql.Query(galaxy=GALAXY, **selected, anywhere=tuple(selected))

    sums = pql.mode_totals(pql.agreements(GALAXY), pql.selection(query), query.anywhere)

    for mode, slots in selected.items():
        gallery = dense(table, mode, pql.KEPT)
        overlaps = gallery @ dense(own, mode, TOP_CODES)[0, list(slots)].T
        if mode in SPECTRUM_MODES:
            overlaps = np.maximum(overlaps, pql.SPECTRUM_TOKEN_FLOOR)
        expected = np.log(overlaps).max(axis=1).sum(axis=1)
        np.testing.assert_allclose(sums[mode], expected, rtol=1e-4, err_msg=mode)


def test_a_position_independent_mode_without_a_selection_is_rejected(
    tree: Path,
) -> None:
    with pytest.raises(ValidationError):
        pql.Query(galaxy=GALAXY, ls_image=(0,), anywhere=("desi_spectrum",))


def test_a_position_independent_map_shows_the_best_selected_slot_at_each_slot(
    table: pa.Table,
) -> None:
    own = table.slice(GALAXY, 1)
    slots = [0, 17, 300]
    query = pql.Query(galaxy=GALAXY, ls_image=tuple(slots), anywhere=("ls_image",))

    results = pql.search(query)

    shown = dense(table.take(results.galaxies), "ls_image", pql.KEPT)
    selected = dense(own, "ls_image", TOP_CODES)[0, slots]
    overlaps = np.log(shown @ selected.T)
    peaks = np.log(selected.max(axis=-1))
    np.testing.assert_allclose(
        results.maps["ls_image"], np.exp((overlaps - peaks).max(axis=-1)), rtol=1e-4
    )
    np.testing.assert_array_equal(
        results.positions["ls_image"], overlaps.argmax(axis=1)
    )


def test_a_cached_matrix_is_served_while_another_galaxy_builds(tree: Path) -> None:
    pql.agreements(GALAXY)
    served = []

    with pql.BUILDING:
        worker = Thread(target=lambda: served.append(pql.agreements(GALAXY)))
        worker.start()
        worker.join(timeout=10)

    assert served


def test_cosines_compare_every_pair_of_the_galaxy_s_slots_of_a_mode(
    table: pa.Table,
) -> None:
    own = table.slice(GALAXY, 1)

    for mode in (*IMAGE_MODES, *SPECTRUM_MODES):
        found = pql.cosines(GALAXY, mode)

        rows = dense(own, mode, TOP_CODES)[0]
        unit = rows / np.linalg.norm(rows, axis=-1, keepdims=True)
        np.testing.assert_allclose(found, unit @ unit.T, atol=1e-5, err_msg=mode)


def partial_correlations(values: np.ndarray, chosen: np.ndarray) -> list[float]:
    score = values[:, chosen].sum(axis=1)
    control = (values.sum(axis=1) - score) / (pql.WIDTH - len(chosen))
    design = np.c_[np.ones(len(values)), control]

    def residual(target: np.ndarray) -> np.ndarray:
        return target - design @ np.linalg.lstsq(design, target, rcond=None)[0]

    return [
        np.corrcoef(residual(score), residual(values[:, column]))[0, 1]
        for column in range(pql.WIDTH)
    ]


def test_saliency_is_the_partial_correlation_over_the_other_galaxies(
    query: pql.Query,
) -> None:
    values = pql.agreements(GALAXY).values.astype(np.float64)
    others = np.delete(values, GALAXY, axis=0)

    expected = partial_correlations(others, pql.columns(pql.selection(query)))

    np.testing.assert_allclose(pql.saliency(query), expected, atol=1e-4)


def test_position_independent_saliency_uses_each_slot_s_best_overlap(
    table: pa.Table, query: pql.Query
) -> None:
    anywhere = query.model_copy(update={"anywhere": ("ls_image",)})
    values = pql.agreements(GALAXY).values.astype(np.float64)
    others = np.delete(np.arange(len(values)), GALAXY)
    gallery = dense(table.take(others), "ls_image", pql.KEPT)
    own = dense(table.slice(GALAXY, 1), "ls_image", TOP_CODES)[0]
    best = np.log(gallery @ own.T).max(axis=1)
    start = pql.OFFSETS["ls_image"]
    values = values[others]
    values[:, start : start + N_IMAGE_TOKENS] = best

    expected = partial_correlations(values, pql.columns(pql.selection(anywhere)))

    np.testing.assert_allclose(pql.saliency(anywhere), expected, atol=1e-4)


def test_saliency_is_finite_and_bounded_when_every_slot_is_selected(
    tree: Path,
) -> None:
    galaxy = int(
        np.flatnonzero(np.logical_and.reduce([pql.observed(c) for c in OBSERVATIONS]))[
            0
        ]
    )
    everything = pql.Selection(
        galaxy=galaxy,
        ls_image=tuple(range(N_IMAGE_TOKENS)),
        hsc_image=tuple(range(N_IMAGE_TOKENS)),
        desi_spectrum=tuple(range(N_SPECTRUM_TOKENS)),
        sdss_spectrum=tuple(range(N_SPECTRUM_TOKENS)),
        table_values=tuple(range(len(pql.TABLE_SLOTS))),
    )

    found = pql.saliency(everything)

    assert np.isfinite(found).all()
    assert (np.abs(found) <= 1).all()


def test_results_open_with_the_query_galaxy_then_the_best_other_galaxies(
    query: pql.Query,
) -> None:
    matches = 4
    ranked = query.model_copy(update={"matches": matches})

    results = pql.search(ranked)

    expected = pql.scores(ranked)
    others = np.delete(np.arange(len(expected)), GALAXY)
    best = others[np.argsort(-expected[others], kind="stable")][:matches]
    assert results.galaxies.tolist() == [GALAXY, *best.tolist()]
    np.testing.assert_array_equal(results.scores, expected[results.galaxies])
    for mode, values in mode_sums(ranked).items():
        np.testing.assert_array_equal(results.sums[mode], values[results.galaxies])
    np.testing.assert_array_equal(
        results.similarity, similarity(ranked)[results.galaxies]
    )


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
    sums = mode_sums(query)

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
    repeated = mode_sums(
        pql.Query(galaxy=GALAXY, ls_image=(7, 7, 9), table_values=(2, 2))
    )
    once = mode_sums(pql.Query(galaxy=GALAXY, ls_image=(7, 9), table_values=(2,)))

    for mode, values in once.items():
        np.testing.assert_array_equal(repeated[mode], values)


def test_similarity_ranks_galaxies_as_the_score_does(query: pql.Query) -> None:
    similarities = similarity(query)

    np.testing.assert_array_equal(
        np.argsort(-similarities, kind="stable"),
        np.argsort(-pql.scores(query), kind="stable"),
    )


def test_one_mode_similarity_is_the_mean_fraction_of_the_best_overlap(
    table: pa.Table,
) -> None:
    single = pql.Query(galaxy=GALAXY, table_values=(2, 6))
    own = dense(table.slice(GALAXY, 1), "ls_table", TOP_CODES)[0, [1, 5]]

    similarities = similarity(single)

    overlaps = log_overlaps(
        "ls_table", dense(table, "ls_table", pql.KEPT)[:, [1, 5]], own
    )
    expected = np.exp((overlaps - np.log(own.max(axis=-1))).mean(axis=1))
    np.testing.assert_allclose(similarities, expected, rtol=1e-4)


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
