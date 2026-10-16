from pathlib import Path

import numpy as np
import pytest
import torch

from app import embeddings, grams
from app.config import GRAM_NAMES, artifact

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


def test_a_galaxy_s_own_row_gives_its_stored_query_vector() -> None:
    gram = torch.from_numpy(
        np.random.default_rng(2).standard_normal((GALAXIES, GALAXIES))
    )
    found = embeddings.embedding(gram)
    chosen = [3, 0, 17]

    np.testing.assert_allclose(
        embeddings.query_vectors(gram[chosen], found),
        found["queries"][chosen],
        atol=1e-9,
    )


def test_a_whole_mode_search_ranks_by_the_mean_standardised_score_of_its_modes(
    tree: Path,
) -> None:
    embeddings.stored.cache_clear()
    query = embeddings.ModeQuery(
        galaxy=4,
        modes=("ls_image", "desi_spectrum", "redshift"),
        anywhere=("desi_spectrum",),
        matches=5,
    )

    found = embeddings.mode_search(query)

    standardised = []
    for name in ("ls_image", "desi_spectrum_anywhere", "redshift"):
        stored = np.load(artifact(f"token_embedding_{name}"))
        cut = slice(embeddings.SEARCH_RANK)
        row = (
            stored["queries"][4, cut].astype(np.float64)
            @ stored["galleries"][:, cut].T.astype(np.float64)
            + stored["offsets"]
        )
        standardised.append((row - row.mean()) / row.std())
    expected = np.mean(standardised, axis=0)
    others = np.argsort(-expected, kind="stable")
    order = np.concatenate(([4], others[others != 4][:5]))
    np.testing.assert_array_equal(found.galaxies, order)
    np.testing.assert_allclose(found.scores, expected[order], atol=1e-3)
    np.testing.assert_allclose(
        sum(found.contributions.values()), found.scores, atol=1e-12
    )
