from pathlib import Path

import numpy as np
import pytest
import torch

from app import embeddings, grams, pql
from app.config import (
    GRAM_NAMES,
    IMAGE_MODES,
    SPECTRUM_MODES,
    TABLE_MODES,
    artifact,
)

GALAXIES = 40
RANK = 5


def test_each_stored_embedding_is_the_best_rank_approximation_of_its_centred_gram(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    monkeypatch.setattr(embeddings, "TOKEN_EMBEDDING_RANK", RANK)

    grams.generate_token_grams()
    embeddings.generate_token_embeddings()

    for name in GRAM_NAMES:
        gram = np.load(artifact(f"token_gram_{name}")).astype(np.float64)
        found = np.load(artifact(f"token_embedding_{name}"))
        centred = (
            gram - gram.mean(axis=1, keepdims=True) - gram.mean(axis=0) + gram.mean()
        )
        residual = centred - found["queries"] @ found["galleries"].T
        singular_values = found["singular_values"]
        np.testing.assert_allclose(
            (residual**2).sum(),
            (singular_values[RANK:] ** 2).sum(),
            rtol=1e-4,
            err_msg=name,
        )
        for part in ("queries", "galleries"):
            np.testing.assert_allclose(
                (found[part] ** 2).sum(axis=0),
                singular_values[:RANK],
                rtol=1e-4,
                err_msg=name,
            )
        np.testing.assert_allclose(
            found["offsets"], gram.mean(axis=0), rtol=1e-5, err_msg=name
        )
        np.testing.assert_array_equal(
            found["self_scores"], np.diagonal(gram), err_msg=name
        )


def test_at_full_rank_scores_rank_galaxies_as_a_one_way_gram_does() -> None:
    gram = torch.from_numpy(
        np.random.default_rng(1).standard_normal((GALAXIES, GALAXIES))
    )

    found = embeddings.embedding(gram)

    difference = found["queries"] @ found["galleries"].T + found["offsets"] - gram
    np.testing.assert_allclose(difference - difference[:, :1], 0, atol=1e-9)


def test_template_embeddings_decompose_each_template_gram_with_widths(
    tree: Path, random_weights: None
) -> None:
    rng = np.random.default_rng(3)
    count = pql.predictions().num_rows
    for name in GRAM_NAMES:
        np.save(
            artifact(f"template_gram_{name}"),
            rng.normal(size=(count, count)).astype(np.float32),
        )
    np.savez(
        artifact("observed_reference"),
        **{
            f"{mode}_depth": rng.random(values)
            for mode, (_, values, _) in TABLE_MODES.items()
        },
        **{f"{mode}_depth": rng.random(1) for mode in (*IMAGE_MODES, *SPECTRUM_MODES)},
    )

    embeddings.generate_template_embeddings()

    for name in GRAM_NAMES:
        gram = np.load(artifact(f"template_gram_{name}")).astype(np.float64)
        found = np.load(artifact(f"template_embedding_{name}"))
        difference = found["queries"] @ found["galleries"].T + found["offsets"] - gram
        np.testing.assert_allclose(
            difference - difference[:, :1], 0, atol=1e-3, err_msg=name
        )
        assert found["widths"].shape == (count,), name
        assert np.isfinite(found["widths"]).all(), name
