"""Properties the fixture tree must hold for conclusions drawn from it to mean anything."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from app.config import ANCHOR, DIM, artifact
from app.search import source


def _anchor_rows(galaxies: int) -> np.ndarray:
    """Every galaxy's anchor cell, as float32 rows."""
    cells = source("encoded").to_table(columns=[ANCHOR]).column(ANCHOR).combine_chunks()
    return np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
        galaxies, -1, DIM
    )


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12)


def test_point_sets_share_one_projection(tree, galaxies: int) -> None:
    """`mean_points` and `full_points` must be coordinates in one space.

    Production gets this for free: one trained projector is applied to both. The
    fixture has to arrange it, and getting it wrong is invisible — the files
    still load and every other test still passes.

    Recovers the projection from `full_points`, then checks `mean_points` agrees
    with it, so nothing here is compared against the fixture's own basis.
    """
    rows = _anchor_rows(galaxies)
    full = pq.read_table(artifact("full_points"))
    mean = pq.read_table(artifact("mean_points"))

    observed = np.column_stack([full.column("x"), full.column("y")])
    basis, *_ = np.linalg.lstsq(_unit(rows.reshape(-1, DIM)), observed, rcond=None)

    expected = _unit(rows.mean(axis=1)) @ basis
    np.testing.assert_allclose(
        np.column_stack([mean.column("x"), mean.column("y")]),
        expected,
        atol=1e-3,
    )


def test_full_points_row_order_follows_galaxy_order(tree, galaxies: int) -> None:
    """`full_points` groups every galaxy's rows together, in row order."""
    owner = np.asarray(pq.read_table(artifact("full_points")).column("galaxy"))

    assert np.all(np.diff(owner) >= 0)
    assert owner[0] == 0
    assert owner[-1] == galaxies - 1
