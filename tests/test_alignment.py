import importlib

import pytest

from app.config import DIM, GEMMA_DIM, device

torch = pytest.importorskip("torch")
alignment = importlib.import_module("app.alignment")

ROWS = 2048


def test_least_squares_recovers_an_affine_map() -> None:
    generator = torch.Generator().manual_seed(0)
    aion = torch.randn(ROWS, DIM, generator=generator)
    weights = torch.randn(DIM + 1, GEMMA_DIM, generator=generator)
    gemma = torch.nn.functional.pad(aion, (0, 1), value=1.0) @ weights

    fitted = alignment.fit_linear(aion, gemma)

    assert torch.allclose(fitted, weights, atol=1e-3)


def test_recall_finds_every_row_when_the_prediction_is_exact() -> None:
    rows = torch.nn.functional.normalize(torch.randn(ROWS, GEMMA_DIM), dim=-1).to(
        device()
    )

    assert alignment.recall(rows, rows) == 1.0
