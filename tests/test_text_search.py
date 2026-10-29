import math
from pathlib import Path

import numpy as np
import pytest
import torch

from app.config import MODES, artifact
from app.text_search import (
    SHARE_NAMES,
    TextQuery,
    breadth,
    cosines,
    point_lengths,
    text_search,
)


def test_a_point_s_cosine_from_its_coefficients_is_that_of_the_document_it_stands_for() -> (
    None
):
    generator = torch.Generator().manual_seed(0)
    components = torch.linalg.qr(torch.randn(12, 4, generator=generator)).Q.T
    centre = torch.randn(12, generator=generator) / 4
    points = torch.randn(3, 5, 4, generator=generator)
    directions = torch.nn.functional.normalize(
        torch.randn(7, 12, generator=generator), dim=-1
    )
    lengths = point_lengths(points, components, centre)

    found = cosines(points, lengths, directions, components, centre)

    documents = centre + points @ components
    expected = torch.einsum(
        "gpd,vd->vgp", torch.nn.functional.normalize(documents, dim=-1), directions
    )
    torch.testing.assert_close(found, expected)


def expected_scores(query: np.ndarray) -> np.ndarray:
    with np.load(artifact("text_bases")) as stored:
        components, centre = stored["components"], stored["centre"]
        kappa = float(stored["kappa"])
    points = np.load(artifact("text_clouds")).astype(np.float64)
    direction = query - np.load(artifact("text_query_mean"))
    direction /= np.linalg.norm(direction)
    documents = centre[:, None, None] + np.einsum("kgpr,krd->kgpd", points, components)
    documents /= np.linalg.norm(documents, axis=-1, keepdims=True)
    logits = kappa * (documents @ direction)
    count = logits.shape[0] * logits.shape[2]
    peak = logits.max(axis=(0, 2))
    summed = np.exp(logits - peak[None, :, None]).sum(axis=(0, 2))
    return peak + np.log(summed) - math.log(count)


def test_a_galaxy_s_score_is_the_query_s_mean_likelihood_over_its_cloud(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(0).standard_normal(768).astype(np.float32)
    monkeypatch.setattr("app.text_search.embed_queries", lambda _: query[None])

    found = text_search(TextQuery(text="a galaxy", matches=5))

    expected = expected_scores(query)
    order = np.argsort(-expected)[:5]
    np.testing.assert_array_equal(found.galaxies.numpy(), order)
    np.testing.assert_allclose(found.scores.numpy(), expected[order], rtol=1e-4)
    probabilities = np.exp(expected - expected.max())
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(
        found.probabilities.numpy(), probabilities[order], rtol=1e-3
    )


def test_shares_split_each_match_over_the_chosen_modes(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(1).standard_normal(768).astype(np.float32)
    monkeypatch.setattr("app.text_search.embed_queries", lambda _: query[None])

    found = text_search(TextQuery(text="a galaxy", modes={"ls_image", "hsc_image"}))

    shares = dict(zip(SHARE_NAMES, found.shares.T))
    torch.testing.assert_close(found.shares.sum(dim=1), torch.ones_like(found.scores))
    assert (shares["combined"] > 0).all()
    for mode in set(MODES) - {"ls_image", "hsc_image"}:
        assert (shares[mode] == 0).all()


def test_a_galaxy_s_score_from_one_mode_counts_no_combined_views(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = np.random.default_rng(2).standard_normal(768).astype(np.float32)
    monkeypatch.setattr("app.text_search.embed_queries", lambda _: query[None])

    found = text_search(TextQuery(text="a galaxy", modes={"desi_spectrum"}))

    torch.testing.assert_close(
        found.shares[:, SHARE_NAMES.index("desi_spectrum")],
        torch.ones_like(found.scores),
    )


def test_breadth_runs_from_one_galaxy_to_every_galaxy() -> None:
    assert breadth(torch.zeros(8)) == pytest.approx(1)
    assert breadth(torch.tensor([1000.0, 0, 0, 0])) == pytest.approx(1 / 4)
