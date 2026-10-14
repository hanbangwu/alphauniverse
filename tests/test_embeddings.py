from pathlib import Path

import numpy as np
import pytest
import torch

from app import embeddings, grams
from app.config import GRAM_NAMES, artifact

GALAXIES = 40
RANK = 5


@pytest.fixture(scope="module")
def stored(tree: Path) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
        patch.setattr(embeddings, "RANK", RANK)
        grams.generate_token_grams()
        embeddings.generate_token_embeddings()
    return {
        name: (
            np.load(artifact(f"token_gram_{name}")),
            dict(np.load(artifact(f"token_embedding_{name}"))),
        )
        for name in GRAM_NAMES
    }


def test_each_stored_embedding_is_the_best_rank_approximation_of_its_centred_gram(
    stored: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
) -> None:
    for name, (gram, found) in stored.items():
        gram = gram.astype(np.float64)
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


def test_at_full_rank_scores_rank_galaxies_as_a_one_way_gram_does() -> None:
    gram = torch.from_numpy(
        np.random.default_rng(1).standard_normal((GALAXIES, GALAXIES))
    )

    found = embeddings.embedding(gram)

    difference = (
        embeddings.scores(found["queries"], found["galleries"], found["offsets"]) - gram
    )
    np.testing.assert_allclose(difference - difference[:, :1], 0, atol=1e-9)
