import importlib
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from app.config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    PREDICTIONS,
    REDSHIFT,
    STORE_COLUMNS,
    TOP_CODES,
    VOCABULARY,
    artifact,
    build_dir,
)

pytest.importorskip("torch")
predictions_module = importlib.import_module("app.predictions")


def column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().flatten())


@pytest.fixture(scope="module")
def store(
    tree: Path, random_weights: None, tmp_path_factory: pytest.TempPathFactory
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    tokens = artifact("tokens")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path_factory.mktemp("store")))
        build_dir().mkdir(parents=True)
        shutil.copy(tokens, artifact("tokens"))
        predictions_module.generate_predictions()
        table = pa.ipc.open_file(pa.memory_map(str(artifact("predictions")))).read_all()
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
    redshift = np.exp(column(table, REDSHIFT).astype(np.float64))
    np.testing.assert_allclose(redshift.reshape(galaxies, -1).sum(-1), 1, atol=0.02)
    for survey, keys in predictions_module.TABLE_VALUES.items():
        kept = np.exp(column(table, f"{survey}_log_probabilities").astype(np.float64))
        tails = np.exp(column(table, f"{survey}_tails").astype(np.float64))
        table_values = np.exp(
            column(table, f"{survey}_table_values").astype(np.float64)
        )
        np.testing.assert_allclose(
            kept.reshape(galaxies, N_IMAGE_TOKENS, -1).sum(-1)
            + tails.reshape(galaxies, N_IMAGE_TOKENS),
            1,
            atol=0.02,
        )
        np.testing.assert_allclose(
            table_values.reshape(galaxies, len(keys), -1).sum(-1), 1, atol=0.02
        )
    for survey in predictions_module.SPECTRA:
        quantised = column(table, f"{survey}_coefficients").reshape(
            galaxies, N_SPECTRUM_TOKENS, -1
        )
        offsets = column(table, f"{survey}_offsets").reshape(
            galaxies, N_SPECTRUM_TOKENS, 1
        )
        steps = column(table, f"{survey}_steps").reshape(galaxies, N_SPECTRUM_TOKENS, 1)
        spectrum_tokens = (
            basis[f"{survey}_mean"]
            + (offsets + quantised * steps) @ basis[f"{survey}_directions"]
        )
        np.testing.assert_allclose(spectrum_tokens.sum(-1), 1, atol=0.02)


def test_kept_image_token_codes_are_the_most_probable_in_descending_order() -> None:
    log_probabilities = np.log(
        np.random.default_rng(1).dirichlet(np.full(IMAGE_VOCABULARY, 0.5), size=3)
    )

    codes, _, _ = predictions_module.image_token_codes(log_probabilities)

    assert np.array_equal(codes, np.argsort(-log_probabilities, axis=1)[:, :TOP_CODES])


def test_spectrum_token_basis_from_moments_matches_a_direct_principal_component_analysis(
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


def test_spectrum_token_coefficients_round_to_within_half_a_step_of_the_projection(
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


@pytest.mark.parametrize("redshift", [None, [7]])
def test_a_redshift_token_joins_the_context_only_where_the_galaxy_has_one(
    redshift: list[int] | None,
) -> None:
    row = {survey: None for survey in STORE_COLUMNS} | {
        ANCHOR: list(range(N_IMAGE_TOKENS + 12)),
        REDSHIFT: redshift,
    }

    tokens = predictions_module.inputs(row)

    found = tokens.get(predictions_module.REDSHIFT_KEY)
    assert (None if found is None else found.tolist()) == (
        None if redshift is None else [redshift]
    )


def test_every_mode_gives_a_normalised_distribution_per_slot_over_its_vocabulary(
    tree: Path, random_weights: None
) -> None:
    row = predictions_module.source("tokens").head(1).to_pylist()[0]

    predicted = predictions_module.predictions(
        predictions_module.inputs(row), predictions_module.TARGETS
    )

    for mode in predictions_module.MODES:
        found = predictions_module.distributions(predicted, mode)
        assert found.shape == (len(mode.keys) * len(mode.positions), mode.vocabulary)
        np.testing.assert_allclose(np.exp(found).sum(axis=-1), 1, rtol=1e-4)


def test_one_batched_decode_matches_decoding_each_block_of_slots_alone(
    tree: Path, random_weights: None
) -> None:
    torch = importlib.import_module("torch")
    encode_module = importlib.import_module("app.encode")
    aion = encode_module.model()
    row = predictions_module.source("tokens").head(1).to_pylist()[0]
    encoded, _, encoder_mask, _ = encode_module.context(predictions_module.inputs(row))

    batched = predictions_module.decode(
        encoded, encoder_mask, predictions_module.TARGETS
    )

    for key, positions in predictions_module.TARGETS.items():
        embedding = aion.decoder_embeddings[key]
        order = torch.randperm(
            len(positions),
            generator=torch.Generator().manual_seed(predictions_module.SEED),
        )
        states = torch.empty(len(positions), encoded.shape[-1])
        for start in range(0, len(positions), predictions_module.CHUNK):
            chosen = order[start : start + predictions_module.CHUNK]
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
                decoded = aion._decode(
                    encoded,
                    encoder_mask,
                    aion.mask_token.expand(1, len(chosen), -1),
                    embedding.pos_emb[:, torch.as_tensor(positions)[chosen]]
                    + embedding.mod_emb,
                    torch.zeros(1, len(chosen), len(chosen), dtype=torch.bool),
                )
            states[chosen] = decoded[0].float()
        with torch.inference_mode():
            alone = torch.log_softmax(embedding.forward_logits(states), dim=-1)
        np.testing.assert_allclose(batched[key], alone.numpy(), atol=2e-3)
