import importlib

import numpy as np
import pytest

from app.config import PREDICTIONS, SPECTRUM_TOKEN_RANK

torch = pytest.importorskip("torch")
predictions_module = importlib.import_module("app.predictions")
compression = importlib.import_module("scripts.benchmarks.compression")
Mode = predictions_module.Mode

SLOTS = 6


def log_probabilities(vocabulary: int, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    drawn = rng.dirichlet(np.full(vocabulary, 0.05), size=(rows, SLOTS))
    return torch.from_numpy(np.log(np.maximum(drawn, 1e-30))).float()


def basis(vocabulary: int) -> tuple:
    rng = np.random.default_rng(1)
    directions, _ = np.linalg.qr(rng.normal(size=(vocabulary, SPECTRUM_TOKEN_RANK)))
    mean = log_probabilities(vocabulary, 1, 0)[0, 0].exp()
    return mean, torch.from_numpy(directions.T).float()


@pytest.mark.parametrize("kind", ["image", "spectrum", "table"])
def test_the_current_overlap_is_the_overlap_of_what_it_decodes(kind: str) -> None:
    mode = Mode(f"ls_{kind}", ("key",), np.arange(SLOTS), 300, "ls")
    scheme = compression.Current(
        mode, *(basis(mode.vocabulary) if kind == "spectrum" else ())
    )
    gallery = log_probabilities(mode.vocabulary, 3, 2)
    query = log_probabilities(mode.vocabulary, 1, 3)[0].exp()

    stored = scheme.encode(gallery)
    torch.testing.assert_close(
        scheme.overlap(stored, query).float(),
        (scheme.decode(stored) * query).sum(dim=-1).float(),
        rtol=1e-4,
        atol=1e-7,
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
        scheme = compression.Current(mode)
        slots = len(mode.keys) * len(mode.positions)
        drawn = np.random.default_rng(5).dirichlet(
            np.full(mode.vocabulary, 0.05), size=(1, slots)
        )
        stored = scheme.encode(torch.from_numpy(np.log(np.maximum(drawn, 1e-30))))
        assert scheme.bits(stored) == widths[mode.name] * 8


def test_exact_log_overlaps_match_float64_where_float32_probabilities_underflow() -> (
    None
):
    quality = importlib.import_module("scripts.benchmarks.compression_quality")
    mode = Mode("ls_table", ("key",), np.arange(SLOTS), 300, "ls")
    gallery = np.full((4, SLOTS, mode.vocabulary), -120.0, dtype=np.float32)
    gallery[np.arange(4), :, np.arange(4)] = 0
    queries = np.asarray([0, 2])

    overlaps = quality.log_overlaps(
        None, mode, gallery, queries, gallery[queries], torch.device("cpu")
    )

    probabilities = np.exp(gallery.astype(np.float64))
    exact = np.log((probabilities[queries, None] * probabilities[None]).sum(axis=-1))
    np.testing.assert_allclose(
        overlaps.found.numpy(), exact.transpose(2, 0, 1), rtol=1e-6
    )
    np.testing.assert_allclose(overlaps.pairs.numpy(), 0, atol=1e-6)
