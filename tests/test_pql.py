from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from app import pql
from app.config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    SPECTRUM_TOKEN_RANK,
    TABLE_VALUE_SURVEYS,
    TOP_CODES,
    VOCABULARY,
)
from app.search import FIRST_HSC_TABLE_VALUE, FIRST_LS_TABLE_VALUE, Query

IMAGE_TOKENS = (0, 17, 300, 575)
SPECTRUM_TOKENS = (40, 41, 42)
TABLE_VALUES = (
    FIRST_LS_TABLE_VALUE + 1,
    FIRST_LS_TABLE_VALUE + 5,
    FIRST_HSC_TABLE_VALUE + 2,
)


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


def dense_table_values(rows: pa.Table, survey: str) -> np.ndarray:
    count = len(TABLE_VALUE_SURVEYS[survey])
    return np.exp(column(rows, f"{survey}_table_values", count, VOCABULARY))


@pytest.fixture(scope="module")
def table(tree: Path) -> pa.Table:
    return pql.predictions()


@pytest.fixture(scope="module")
def query(tree: Path) -> Query:
    return Query(
        galaxy=0,
        image_tokens=IMAGE_TOKENS,
        spectrum_tokens=SPECTRUM_TOKENS,
        table_values=TABLE_VALUES,
    )


def test_mode_sums_equal_a_brute_force_computation_on_dense_distributions(
    table: pa.Table, query: Query
) -> None:
    survey = pql.spectrum_survey(query.galaxy)
    own = table.slice(query.galaxy, 1)
    spectrum_tokens = dense_spectrum_tokens(table, survey)
    expected = {
        f"{ANCHOR}_image": np.einsum(
            "gsv,sv->gs",
            dense_image_tokens(table, ANCHOR, pql.KEPT)[:, IMAGE_TOKENS],
            dense_image_tokens(own, ANCHOR, TOP_CODES)[0, IMAGE_TOKENS],
        ),
        f"{survey}_spectrum": np.maximum(
            np.einsum(
                "gsv,sv->gs",
                spectrum_tokens[:, SPECTRUM_TOKENS],
                spectrum_tokens[query.galaxy, SPECTRUM_TOKENS],
            ),
            pql.SPECTRUM_TOKEN_FLOOR,
        ),
        f"{ANCHOR}_table": np.einsum(
            "gsv,sv->gs",
            dense_table_values(table, ANCHOR)[:, [1, 5]],
            dense_table_values(own, ANCHOR)[0, [1, 5]],
        ),
        "hsc_table": np.einsum(
            "gsv,sv->gs",
            dense_table_values(table, "hsc")[:, [2]],
            dense_table_values(own, "hsc")[0, [2]],
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
    spectrum_tokens = dense_spectrum_tokens(table, survey)

    image_token_maps, spectrum_token_maps, table_values = pql.maps(query, galaxies)

    np.testing.assert_allclose(
        image_token_maps,
        np.log(
            dense_image_tokens(shown, ANCHOR, pql.KEPT)
            @ dense_image_tokens(own, ANCHOR, TOP_CODES)[0, IMAGE_TOKENS].mean(axis=0)
        ),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        spectrum_token_maps,
        np.log(
            np.maximum(
                spectrum_tokens[galaxies]
                @ spectrum_tokens[query.galaxy, SPECTRUM_TOKENS].mean(axis=0),
                pql.SPECTRUM_TOKEN_FLOOR,
            )
        ),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        table_values,
        np.log(
            np.hstack(
                [
                    (
                        dense_table_values(shown, catalogue)
                        * dense_table_values(own, catalogue)
                    ).sum(axis=-1)
                    for catalogue in TABLE_VALUE_SURVEYS
                ]
            )
        ),
        rtol=1e-4,
    )


def test_maps_of_unselected_modes_are_missing(tree: Path) -> None:
    image_tokens, spectrum_tokens, _ = pql.maps(
        Query(galaxy=0, table_values=(FIRST_LS_TABLE_VALUE,)), np.asarray([3])
    )

    assert np.isnan(image_tokens).all()
    assert np.isnan(spectrum_tokens).all()


def test_a_selection_across_modes_ranks_by_its_mean_standardised_mode_sum(
    query: Query,
) -> None:
    sums = pql.parts(query)

    expected = np.mean(
        [(values - values.mean()) / values.std() for values in sums.values()],
        axis=0,
    )
    np.testing.assert_allclose(pql.scores(query), expected)
