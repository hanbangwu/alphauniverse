import math
from typing import NamedTuple

import numpy as np

from .config import GRID, N_SPECTRUM_TOKENS, SPECTRUM_ORIGIN, SPECTRUM_TOKEN_WIDTH

DECIMALS = 9


class Poses(NamedTuple):
    indices: np.ndarray
    weights: np.ndarray
    parameters: np.ndarray


def orientations(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(offsets, axis=1).max()
    turns = max(1, 4 * math.ceil(math.pi * radius))
    angles = 2 * np.pi * np.arange(turns) / turns
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    turned = []
    for mirror in (1, -1):
        rows, columns = offsets[:, 0], mirror * offsets[:, 1]
        turned.append(
            np.stack((cos * rows - sin * columns, sin * rows + cos * columns), axis=-1)
        )
    parameters = np.stack((np.repeat([0, 1], turns), np.tile(angles, 2)), axis=-1)
    return np.concatenate(turned), parameters


def shifts(placed: np.ndarray) -> np.ndarray:
    low = np.ceil(-placed.min(axis=0)).astype(int)
    high = np.floor(GRID - 1 - placed.max(axis=0)).astype(int)
    rows, columns = np.meshgrid(
        np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1), indexing="ij"
    )
    return np.stack((rows.ravel(), columns.ravel()), axis=-1)


def bilinear(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.floor(positions).astype(int)
    fraction = positions - start
    indices, weights = [], []
    for row in (0, 1):
        for column in (0, 1):
            cell = np.minimum(start + (row, column), GRID - 1)
            indices.append(cell[..., 0] * GRID + cell[..., 1])
            weights.append(
                np.abs(1 - row - fraction[..., 0])
                * np.abs(1 - column - fraction[..., 1])
            )
    return np.stack(indices, axis=-1), np.stack(weights, axis=-1).astype(np.float32)


def image_poses(slots: np.ndarray) -> Poses:
    points = np.stack(np.divmod(slots, GRID), axis=-1).astype(float)
    centre = np.round(points.mean(axis=0))
    turned, turns = orientations(points - centre)
    positions, parameters = [], []
    for placed, turn in zip(np.round(turned + centre, DECIMALS), turns, strict=True):
        moves = shifts(placed)
        positions.append(placed + moves[:, None])
        parameters.append(np.c_[np.tile(turn, (len(moves), 1)), moves])
    positions = np.concatenate(positions)
    _, first = np.unique(
        positions.reshape(len(positions), -1), axis=0, return_index=True
    )
    first.sort()
    return Poses(*bilinear(positions[first]), np.concatenate(parameters)[first])


def window(wavelengths: np.ndarray) -> np.ndarray:
    return np.round((wavelengths - SPECTRUM_ORIGIN) / SPECTRUM_TOKEN_WIDTH, DECIMALS)


def spectrum_poses(slots: np.ndarray) -> Poses:
    low = SPECTRUM_ORIGIN + SPECTRUM_TOKEN_WIDTH * slots
    high = low + SPECTRUM_TOKEN_WIDTH
    end = SPECTRUM_ORIGIN + SPECTRUM_TOKEN_WIDTH * N_SPECTRUM_TOKENS
    step = SPECTRUM_TOKEN_WIDTH / (2 * high.max())
    steps = np.arange(
        math.ceil(math.log(SPECTRUM_ORIGIN / low.min()) / step),
        math.floor(math.log(end / high.max()) / step) + 1,
    )
    stretches = np.exp(steps * step)[:, None]
    starts, stops = window(stretches * low), window(stretches * high)
    first = np.floor(starts).astype(int)
    reads = first[..., None] + np.arange(math.ceil(stretches.max()) + 1)
    covered = np.minimum(stops[..., None], reads + 1) - np.maximum(
        starts[..., None], reads
    )
    weights = np.maximum(covered, 0) / (stops - starts)[..., None]
    return Poses(
        np.minimum(reads, N_SPECTRUM_TOKENS - 1),
        weights.astype(np.float32),
        stretches,
    )
