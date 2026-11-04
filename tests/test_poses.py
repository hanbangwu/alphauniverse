import numpy as np

from app.config import GRID, N_SPECTRUM_TOKENS, SPECTRUM_ORIGIN, SPECTRUM_TOKEN_WIDTH
from app.poses import image_poses, spectrum_poses, window

SLOTS = np.array([0, 2, 50, 99])


def placed(indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cells = np.stack(np.divmod(indices, GRID), axis=-1)
    return (cells * weights[..., None]).sum(axis=-2)


def distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[..., :, None, :] - points[..., None, :, :], axis=-1)


def test_every_image_pose_moves_the_selection_rigidly() -> None:
    found = image_poses(SLOTS)

    moved = placed(found.indices, found.weights)

    cells = np.stack(np.divmod(SLOTS, GRID), axis=-1).astype(float)
    np.testing.assert_allclose(found.weights.sum(axis=-1), 1, rtol=1e-6)
    np.testing.assert_allclose(
        distances(moved),
        np.broadcast_to(distances(cells), (len(moved), 4, 4)),
        atol=1e-5,
    )


def test_quarter_turns_land_on_cells_and_the_identity_is_a_pose() -> None:
    found = image_poses(SLOTS)

    quarter = np.isclose(np.mod(found.parameters[:, 1], np.pi / 2), 0)

    assert quarter.sum() > 8
    assert np.isin(found.weights[quarter], (0, 1)).all()
    identity = (found.indices[..., 0] == SLOTS).all(axis=1) & (
        found.weights[..., 0] == 1
    ).all(axis=1)
    assert identity.sum() == 1


def test_image_angles_move_the_farthest_slot_at_most_half_a_cell() -> None:
    slots = np.array([275, 278, 323])

    found = image_poses(slots)

    angles = np.unique(found.parameters[:, 1])
    cells = np.stack(np.divmod(slots, GRID), axis=-1)
    radius = np.linalg.norm(cells - np.round(cells.mean(axis=0)), axis=-1).max()
    assert np.diff(np.append(angles, 2 * np.pi)).max() * radius <= 0.5


def test_spectrum_poses_read_each_stretched_window_by_the_share_it_covers() -> None:
    slots = np.array([40, 41, 42, 120])

    found = spectrum_poses(slots)

    stretches = found.parameters
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
