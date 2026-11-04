import numpy as np
import pytest

from app.config import GRID, N_SPECTRUM_TOKENS, SPECTRUM_ORIGIN, SPECTRUM_TOKEN_WIDTH
from app.poses import image_poses, spectrum_poses, window

SLOTS = np.array([0, 2, 50, 99])


def exact_positions(slots: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    cells = np.stack(np.divmod(slots, GRID), axis=-1).astype(float)
    centre = np.round(cells.mean(axis=0))
    mirror, angle, shift = parameters[:, :1], parameters[:, 1:2], parameters[:, 2:]
    rows = cells[:, 0] - centre[0]
    columns = np.where(mirror, -1, 1) * (cells[:, 1] - centre[1])
    turned = np.stack(
        (
            np.cos(angle) * rows - np.sin(angle) * columns,
            np.sin(angle) * rows + np.cos(angle) * columns,
        ),
        axis=-1,
    )
    return turned + centre + shift[:, None]


@pytest.mark.parametrize("step", [1.0, 0.5])
def test_every_image_pose_is_a_rigid_motion_snapped_to_quarter_cells(
    step: float,
) -> None:
    found = image_poses(SLOTS, step)

    positions = found.quarters / 4

    exact = exact_positions(SLOTS, found.parameters)
    assert np.abs(positions - exact).max() <= 1 / 8 + 1e-9
    assert (positions >= 0).all() and (positions <= GRID - 1).all()
    assert len(np.unique(found.quarters.reshape(len(positions), -1), axis=0)) == len(
        positions
    )


def test_quarter_turns_land_on_cells_and_the_identity_is_a_pose() -> None:
    found = image_poses(SLOTS, 1.0)

    quarter = np.isclose(np.mod(found.parameters[:, 1], np.pi / 2), 0)

    assert quarter.sum() > 8
    assert (found.quarters[quarter] % 4 == 0).all()
    cells = np.stack(np.divmod(SLOTS, GRID), axis=-1) * 4
    assert (found.quarters == cells).all(axis=(1, 2)).sum() == 1


def test_image_angles_move_the_farthest_slot_at_most_half_a_cell() -> None:
    slots = np.array([275, 278, 323])

    found = image_poses(slots, 1.0)

    angles = np.unique(found.parameters[:, 1])
    cells = np.stack(np.divmod(slots, GRID), axis=-1)
    radius = np.linalg.norm(cells - np.round(cells.mean(axis=0)), axis=-1).max()
    assert np.diff(np.append(angles, 2 * np.pi)).max() * radius <= 0.5


def test_half_cell_shifts_keep_every_whole_cell_pose_and_add_more() -> None:
    cell, half = image_poses(SLOTS, 1.0), image_poses(SLOTS, 0.5)

    found = {pose.tobytes() for pose in half.quarters}

    assert all(pose.tobytes() in found for pose in cell.quarters)
    assert len(half.quarters) > len(cell.quarters)


def test_spectrum_poses_read_each_stretched_window_by_the_share_it_covers() -> None:
    slots = np.array([40, 41, 42, 120])

    found = spectrum_poses(slots)

    stretches = found.stretches
    starts = window(stretches * (SPECTRUM_ORIGIN + SPECTRUM_TOKEN_WIDTH * slots))
    samples = starts[..., None] + stretches[..., None] * (np.arange(4000) + 0.5) / 4000
    expected = np.zeros((*starts.shape, N_SPECTRUM_TOKENS))
    np.put_along_axis(expected, found.indices, found.weights, axis=-1)
    shares = np.stack(
        [
            np.bincount(row, minlength=N_SPECTRUM_TOKENS) / len(row)
            for row in np.floor(samples).astype(int).reshape(-1, samples.shape[-1])
        ]
    ).reshape(expected.shape)
    np.testing.assert_allclose(expected, shares, atol=1e-3)
    reddest = SPECTRUM_ORIGIN + SPECTRUM_TOKEN_WIDTH * (slots.max() + 1)
    steps = np.diff(stretches[:, 0]) * reddest / SPECTRUM_TOKEN_WIDTH
    assert steps.max() <= 0.5 * stretches.max()
    identity = stretches[:, 0] == 1
    assert identity.sum() == 1
    np.testing.assert_array_equal(found.indices[identity][0, :, 0], slots)
