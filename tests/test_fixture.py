"""Properties of the fixture tree that the other tests assume."""

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize

from app.config import ANCHOR, DIM, artifact
from app.search import source


def test_point_sets_share_one_projection(tree, galaxies: int) -> None:
    """`mean_points` and `full_points` must be coordinates in one space.

    The fixture has to arrange what one trained projector guarantees in
    production, and nothing else surfaces a mistake: the files still load and
    every other test still passes. The projection is recovered from
    `full_points`, so nothing is compared against the fixture's own basis.
    """
    cells = source("encoded").to_table(columns=[ANCHOR]).column(ANCHOR).combine_chunks()
    rows = np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
        galaxies, -1, DIM
    )
    full = pq.read_table(artifact("full_points"))
    mean = pq.read_table(artifact("mean_points"))

    observed = np.column_stack([full.column("x"), full.column("y")])
    basis, *_ = np.linalg.lstsq(normalize(rows.reshape(-1, DIM)), observed)

    expected = normalize(rows.mean(axis=1)) @ basis
    np.testing.assert_allclose(
        np.column_stack([mean.column("x"), mean.column("y")]),
        expected,
        atol=1e-3,
    )


def test_fixture_groups_full_points_by_galaxy(tree, galaxies: int) -> None:
    """The fixture writes `full_points` grouped by galaxy.

    `test_point_sets_share_one_projection` assumes it. Production's artifact
    is not grouped, so this is a property of the fixture alone.
    """
    owner = np.asarray(pq.read_table(artifact("full_points")).column("galaxy"))

    assert np.all(np.diff(owner) >= 0)
    assert owner[0] == 0
    assert owner[-1] == galaxies - 1
