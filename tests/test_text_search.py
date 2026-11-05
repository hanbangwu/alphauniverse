from pathlib import Path

import numpy as np
import pytest
import torch

from app.text_search import TextQuery, text_search
from core.artifacts import artifact
from core.config import QUERY_PREFIX, WEMM_DIM
from core.text_model import embed_queries


def test_galaxies_rank_by_their_mean_cosine_and_each_mode_s_contribution_sums_to_its_lead(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(0).standard_normal(WEMM_DIM).astype(np.float32)
    found = text_search(TextQuery(text="a galaxy", matches=5), lambda _: query[None])

    modes = np.load(artifact("text_vectors")).astype(np.float64)
    direction = query - np.load(artifact("text_query_mean"))
    direction /= np.linalg.norm(direction)
    expected = modes.sum(axis=1) @ direction
    order = np.argsort(-expected)[:5]
    np.testing.assert_array_equal(found.galaxies.numpy(), order)
    np.testing.assert_allclose(found.scores.numpy(), expected[order], rtol=1e-4)
    torch.testing.assert_close(
        found.contributions.sum(dim=1),
        found.scores - float(expected.mean()),
        rtol=1e-4,
        atol=1e-5,
    )


def test_every_query_is_embedded_after_the_search_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    class Model:
        def encode_query(self, texts, normalize_embeddings):
            seen.extend(texts)
            return np.zeros((len(texts), WEMM_DIM), dtype=np.float32)

    monkeypatch.setattr("core.text_model.text_model", Model)

    embed_queries(["spirals", "a red galaxy"])

    assert seen == [QUERY_PREFIX.format("spirals"), QUERY_PREFIX.format("a red galaxy")]
