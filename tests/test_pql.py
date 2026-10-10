from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from app import pql
from app.config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_PATCHES,
    N_SPANS,
    SCALAR_SURVEYS,
    SPAN_RANK,
    TOP_CODES,
    VOCABULARY,
)
from app.search import N_LS_SCALARS, Query

PATCHES = (0, 17, 300, 575)
SPANS = (40, 41, 42)
SCALARS = (1, 5, N_LS_SCALARS + 2)


def column(rows: pa.Table, name: str, *shape: int) -> np.ndarray:
    values = rows[name].combine_chunks().flatten().to_numpy()
    return values.astype(np.float64).reshape(rows.num_rows, *shape)


def dense_cells(rows: pa.Table, survey: str, kept: int) -> np.ndarray:
    codes = column(rows, f"{survey}_codes", N_PATCHES, TOP_CODES).astype(np.int64)
    probabilities = np.exp(
        column(rows, f"{survey}_log_probabilities", N_PATCHES, TOP_CODES)
    )
    rest = np.exp(column(rows, f"{survey}_tails", N_PATCHES)) + probabilities[
        ..., kept:
    ].sum(axis=-1)
    dense = np.repeat(
        (rest / (IMAGE_VOCABULARY - kept))[..., None], IMAGE_VOCABULARY, axis=-1
    )
    np.put_along_axis(dense, codes[..., :kept], probabilities[..., :kept], axis=-1)
    return dense


def dense_spans(rows: pa.Table, survey: str) -> np.ndarray:
    mean, directions, _, _ = pql.basis()[survey]
    coefficients = column(rows, f"{survey}_offsets", N_SPANS, 1) + column(
        rows, f"{survey}_coefficients", N_SPANS, SPAN_RANK
    ) * column(rows, f"{survey}_steps", N_SPANS, 1)
    return mean + coefficients @ directions


def dense_scalars(rows: pa.Table, survey: str) -> np.ndarray:
    count = len(SCALAR_SURVEYS[survey])
    return np.exp(column(rows, f"{survey}_scalars", count, VOCABULARY))


@pytest.fixture(scope="module")
def table(tree: Path) -> pa.Table:
    return pql.predictions()


@pytest.fixture(scope="module")
def query(tree: Path) -> Query:
    return Query(galaxy=0, patches=PATCHES, spans=SPANS, scalars=SCALARS)


def test_mode_sums_equal_a_brute_force_computation_on_dense_distributions(
    table: pa.Table, query: Query
) -> None:
    survey = pql.spectrum_survey(query.galaxy)
    own = table.slice(query.galaxy, 1)
    spans = dense_spans(table, survey)
    expected = {
        f"{ANCHOR}_cells": np.einsum(
            "gsv,sv->gs",
            dense_cells(table, ANCHOR, pql.KEPT)[:, PATCHES],
            dense_cells(own, ANCHOR, TOP_CODES)[0, PATCHES],
        ),
        f"{survey}_spans": np.maximum(
            np.einsum("gsv,sv->gs", spans[:, SPANS], spans[query.galaxy, SPANS]),
            pql.SPAN_FLOOR,
        ),
        f"{ANCHOR}_scalars": np.einsum(
            "gsv,sv->gs",
            dense_scalars(table, ANCHOR)[:, [1, 5]],
            dense_scalars(own, ANCHOR)[0, [1, 5]],
        ),
        "hsc_scalars": np.einsum(
            "gsv,sv->gs",
            dense_scalars(table, "hsc")[:, [2]],
            dense_scalars(own, "hsc")[0, [2]],
        ),
    }

    sums = pql.parts(query)

    assert sums.keys() == expected.keys()
    for mode, overlaps in expected.items():
        np.testing.assert_allclose(sums[mode], np.log(overlaps).sum(axis=1), rtol=1e-4)


def test_maps_overlap_every_slot_with_the_mean_of_the_selected_query_slots(
    table: pa.Table, query: Query
) -> None:
    survey = pql.spectrum_survey(query.galaxy)
    galaxies = np.asarray([3, 0])
    own = table.slice(query.galaxy, 1)
    shown = table.take(galaxies)
    spans = dense_spans(table, survey)

    cells, span_maps, scalars = pql.maps(query, galaxies)

    np.testing.assert_allclose(
        cells,
        np.log(
            dense_cells(shown, ANCHOR, pql.KEPT)
            @ dense_cells(own, ANCHOR, TOP_CODES)[0, PATCHES].mean(axis=0)
        ),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        span_maps,
        np.log(
            np.maximum(
                spans[galaxies] @ spans[query.galaxy, SPANS].mean(axis=0),
                pql.SPAN_FLOOR,
            )
        ),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        scalars,
        np.log(
            np.hstack(
                [
                    (
                        dense_scalars(shown, catalogue) * dense_scalars(own, catalogue)
                    ).sum(axis=-1)
                    for catalogue in SCALAR_SURVEYS
                ]
            )
        ),
        rtol=1e-4,
    )


def test_maps_of_unselected_modes_are_missing(tree: Path) -> None:
    cells, spans, _ = pql.maps(Query(galaxy=0, scalars=(1,)), np.asarray([3]))

    assert np.isnan(cells).all()
    assert np.isnan(spans).all()


def test_a_selection_across_modes_ranks_by_its_mean_standardised_mode_sum(
    query: Query,
) -> None:
    sums = pql.parts(query)

    expected = np.mean(
        [(values - values.mean()) / values.std() for values in sums.values()],
        axis=0,
    )
    np.testing.assert_allclose(pql.scores(query), expected)
