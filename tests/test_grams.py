from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from app import grams, pql
from app.config import N_IMAGE_TOKENS, TokenMode, artifact

GALAXY = 3


def test_a_row_with_search_s_representation_equals_search_with_every_slot_selected(
    tree: Path,
) -> None:
    selected = {mode: np.arange(slots) for mode, slots in pql.WIDTHS.items()}
    positional = {mode: selected[mode] for mode in get_args(TokenMode)}
    anywhere = pql.mode_totals(GALAXY, selected, positional)
    expected = pql.mode_totals(GALAXY, selected, {}) | {
        f"{mode}_anywhere": anywhere[mode] for mode in positional
    }

    for mode in pql.MODES:
        query = grams.forms(mode).rows(slice(GALAXY, GALAXY + 1))
        gallery = grams.forms(mode, pql.top_image_tokens)
        found = grams.scores(query, gallery.rows(slice(None)), gallery)
        for name, row in zip(grams.blocks(mode), found):
            np.testing.assert_allclose(row[0], expected[name], rtol=1e-6, err_msg=name)


def test_each_stored_gram_holds_every_galaxy_s_row_across_tiles(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grams, "ROWS", 5 * N_IMAGE_TOKENS)

    grams.generate_token_grams()

    for mode in pql.MODES:
        forms = grams.forms(mode)
        everyone = forms.rows(slice(None))
        found = grams.scores(everyone, everyone, forms)
        for name, expected in zip(grams.blocks(mode), found):
            np.testing.assert_allclose(
                np.load(artifact(f"token_gram_{name}")),
                expected,
                rtol=1e-6,
                err_msg=name,
            )
