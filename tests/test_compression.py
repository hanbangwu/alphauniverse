import importlib

import numpy as np
import pytest

from app.config import PREDICTIONS

torch = pytest.importorskip("torch")
predictions_module = importlib.import_module("app.predictions")
compression = importlib.import_module("scripts.benchmarks.compression")
Mode = predictions_module.Mode

SLOTS = 6


def log_probabilities(vocabulary: int, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    drawn = rng.dirichlet(np.full(vocabulary, 0.05), size=(rows, SLOTS))
    return torch.from_numpy(np.log(np.maximum(drawn, 1e-30))).float()


def fitted(mode):
    basis = compression.Basis(max(compression.RANKS))
    basis.fit([log_probabilities(mode.vocabulary, 120, 1)])
    return basis


@pytest.mark.parametrize("kind", ["image", "spectrum", "table"])
def test_each_schemes_overlap_is_the_overlap_of_what_it_decodes(kind: str) -> None:
    mode = Mode(f"ls_{kind}", ("key",), np.arange(SLOTS), 300, "ls")
    basis = fitted(mode)
    gallery = log_probabilities(mode.vocabulary, 3, 2)
    query = log_probabilities(mode.vocabulary, 1, 3)[0].exp()

    for scheme in compression.schemes(mode, basis):
        stored = scheme.encode(gallery)
        torch.testing.assert_close(
            scheme.overlap(stored, query).float(),
            (scheme.decode(stored) * query).sum(dim=-1).float(),
            rtol=1e-4,
            atol=1e-7,
            msg=scheme.name,
        )


def test_lossless_schemes_decode_the_distribution() -> None:
    vocabulary = 256
    mode = Mode("ls_spectrum", ("key",), np.arange(SLOTS), vocabulary, "ls")
    basis = fitted(mode)
    gallery = log_probabilities(vocabulary, 3, 4)
    lossless = [
        compression.Dense(compression.FLOATS[0]),
        compression.TopK(vocabulary, "spread", compression.FLOATS[0], vocabulary),
        compression.Projection(vocabulary, compression.FLOATS[0], basis),
    ]

    for scheme in lossless:
        torch.testing.assert_close(
            scheme.decode(scheme.encode(gallery)),
            gallery.exp(),
            rtol=0,
            atol=1e-5,
            msg=scheme.name,
        )


def test_the_current_scheme_takes_the_bytes_the_store_gives_each_mode() -> None:
    columns = {
        field.name: field.type.list_size
        for field in PREDICTIONS
        if field.name != "galaxy"
    }
    widths = {
        "ls_image": columns["ls_codes"] * 2
        + columns["ls_log_probabilities"] * 2
        + columns["ls_tails"] * 4,
        "ls_table": columns["ls_table_values"] * 2,
        "redshift": columns["redshift"] * 2,
    }

    for mode in predictions_module.MODES:
        if mode.name not in widths:
            continue
        scheme = compression.Current(mode, compression.Basis(1))
        slots = len(mode.keys) * len(mode.positions)
        drawn = np.random.default_rng(5).dirichlet(
            np.full(mode.vocabulary, 0.05), size=(1, slots)
        )
        stored = scheme.encode(torch.from_numpy(np.log(np.maximum(drawn, 1e-30))))
        assert scheme.bits(stored) == widths[mode.name] * 8


def test_quantised_values_are_within_half_a_step() -> None:
    values = torch.from_numpy(np.random.default_rng(6).normal(size=(4, 50))).float()

    for precision in compression.VALUES[3:]:
        stored = compression.store(values, precision)
        error = (compression.load(stored) - values).abs()
        assert bool((error <= stored["steps"] * 0.5001).all())


def test_top_p_keeps_the_fewest_codes_holding_the_mass() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    scheme = compression.TopP(0.9, "spread", compression.FLOATS[0], 4)

    indices, mask = scheme.kept(probabilities.log())

    assert indices[mask].tolist() == [0, 1, 2]


def test_dense_log_overlaps_match_float64_where_float32_probabilities_underflow() -> (
    None
):
    quality = importlib.import_module("scripts.benchmarks.compression_quality")
    mode = Mode("ls_table", ("key",), np.arange(SLOTS), 300, "ls")
    gallery = np.full((4, SLOTS, mode.vocabulary), -120.0, dtype=np.float32)
    gallery[np.arange(4), :, np.arange(4)] = 0
    queries = np.asarray([0, 2])

    found, pairs, _, floored = quality.log_overlaps(
        None, mode, gallery, queries, gallery[queries], torch.device("cpu")
    )

    probabilities = np.exp(gallery.astype(np.float64))
    exact = np.log((probabilities[queries, None] * probabilities[None]).sum(axis=-1))
    np.testing.assert_allclose(
        found["exact"].numpy(), exact.transpose(2, 0, 1), rtol=1e-6
    )
    np.testing.assert_allclose(pairs["exact"].numpy(), 0, atol=1e-6)
    assert floored == 0
