import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from app.config import (
    N_PATCHES,
    N_SPANS,
    PREDICTIONS,
    TOP_CODES,
    VOCABULARY,
    artifact,
)

pytest.importorskip("torch")
predictions_module = importlib.import_module("app.predictions")


def column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().flatten())


@pytest.fixture(scope="module")
def store(tree: Path, random_weights: None) -> tuple[pa.Table, dict[str, np.ndarray]]:
    predictions_module.generate_predictions()
    with pa.memory_map(str(artifact("predictions"))) as source:
        table = pa.ipc.open_file(source).read_all()
    with np.load(artifact("prediction_basis")) as basis:
        return table, dict(basis)


@pytest.fixture(scope="module")
def distributions() -> np.ndarray:
    return np.random.default_rng(0).dirichlet(np.full(VOCABULARY, 0.1), size=600)


def test_every_galaxy_has_normalised_predictions_at_every_slot_in_galaxy_order(
    store: tuple[pa.Table, dict[str, np.ndarray]], galaxies: int
) -> None:
    table, basis = store

    assert table.schema.equals(PREDICTIONS)
    assert table["galaxy"].to_pylist() == list(range(galaxies))
    for survey, keys in predictions_module.SCALARS.items():
        kept = np.exp(column(table, f"{survey}_log_probabilities").astype(np.float64))
        tails = np.exp(column(table, f"{survey}_tails").astype(np.float64))
        scalars = np.exp(column(table, f"{survey}_scalars").astype(np.float64))
        np.testing.assert_allclose(
            kept.reshape(galaxies, N_PATCHES, -1).sum(-1)
            + tails.reshape(galaxies, N_PATCHES),
            1,
            atol=0.02,
        )
        np.testing.assert_allclose(
            scalars.reshape(galaxies, len(keys), -1).sum(-1), 1, atol=0.02
        )
    for survey in predictions_module.SPECTRA:
        quantised = column(table, f"{survey}_coefficients").reshape(
            galaxies, N_SPANS, -1
        )
        offsets = column(table, f"{survey}_offsets").reshape(galaxies, N_SPANS, 1)
        steps = column(table, f"{survey}_steps").reshape(galaxies, N_SPANS, 1)
        spans = (
            basis[f"{survey}_mean"]
            + (offsets + quantised * steps) @ basis[f"{survey}_directions"]
        )
        np.testing.assert_allclose(spans.sum(-1), 1, atol=0.02)


def test_kept_cell_codes_are_the_most_probable_in_descending_order() -> None:
    log_probabilities = np.log(
        np.random.default_rng(1).dirichlet(np.full(4375, 0.5), size=3)
    )

    codes, _, _ = predictions_module.cells(log_probabilities)

    assert np.array_equal(codes, np.argsort(-log_probabilities, axis=1)[:, :TOP_CODES])


def test_span_basis_from_moments_matches_a_direct_principal_component_analysis(
    distributions: np.ndarray,
) -> None:
    moments = predictions_module.Moments()
    moments.add(distributions[:250])
    moments.add(distributions[250:])

    mean, directions = moments.basis()

    _, _, reference = np.linalg.svd(
        distributions - distributions.mean(axis=0), full_matrices=False
    )
    np.testing.assert_allclose(mean, distributions.mean(axis=0))
    np.testing.assert_allclose(
        np.abs(np.sum(directions[:10] * reference[:10], axis=1)), 1, atol=1e-6
    )


def test_span_coefficients_round_to_within_half_a_step_of_the_projection(
    distributions: np.ndarray,
) -> None:
    moments = predictions_module.Moments()
    moments.add(distributions)
    mean, directions = moments.basis()

    quantised, offsets, steps = predictions_module.coefficients(
        distributions[:5], mean, directions
    )

    projected = (distributions[:5] - mean) @ directions.T
    error = np.abs(offsets[:, None] + quantised * steps[:, None] - projected)
    assert np.all(error <= steps[:, None] * 0.5001)
