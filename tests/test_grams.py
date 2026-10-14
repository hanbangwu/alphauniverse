from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch

from app import grams, pql
from app.config import N_IMAGE_TOKENS, TOKEN_GRAMS, TokenMode, artifact

GALAXY = 3


def test_a_row_with_search_s_representation_equals_search_with_every_slot_selected(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = {mode: np.arange(slots) for mode, slots in pql.WIDTHS.items()}
    positional = pql.positional_slots(selected, get_args(TokenMode))
    aligned = pql.mode_totals(GALAXY, selected, {})
    anywhere = pql.mode_totals(GALAXY, selected, positional)
    queries = {
        mode: grams.forms(mode).dense(slice(GALAXY, GALAXY + 1)) for mode in TOKEN_GRAMS
    }
    monkeypatch.setattr(pql, "query_image_tokens", pql.top_image_tokens)

    for mode, names in TOKEN_GRAMS.items():
        gallery = grams.forms(mode)
        found = grams.scores(queries[mode], gallery.dense(slice(None)), gallery)
        for name, row, expected in zip(names, found, (aligned[mode], anywhere[mode])):
            np.testing.assert_allclose(row[0], expected, rtol=1e-6, err_msg=name)


def test_each_stored_gram_holds_every_galaxy_s_row_across_tiles(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grams, "ROWS", 5 * N_IMAGE_TOKENS)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")

    grams.generate_token_grams()

    for mode, names in TOKEN_GRAMS.items():
        forms = grams.forms(mode)
        everyone = forms.dense(slice(None))
        for name, expected in zip(names, grams.scores(everyone, everyone, forms)):
            np.testing.assert_allclose(
                np.load(artifact(f"token_gram_{name}")),
                expected,
                rtol=1e-6,
                err_msg=name,
            )
