import math
from typing import NamedTuple

import numpy as np

from .config import GRID, N_SPECTRUM_TOKENS, SPECTRUM_ORIGIN, SPECTRUM_TOKEN_WIDTH

DECIMALS = 9
QUARTERS = 4
LAST = (GRID - 1) * QUARTERS


class ImagePoses(NamedTuple):
    quarters: np.ndarray
    parameters: np.ndarray


class SpectrumPoses(NamedTuple):
    indices: np.ndarray
    weights: np.ndarray
    stretches: np.ndarray


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


def image_poses(slots: np.ndarray, step: float) -> ImagePoses:
    points = np.stack(np.divmod(slots, GRID), axis=-1)
    centre = np.round(points.mean(axis=0))
    turned, turns = orientations(points - centre)
    placed = np.rint((turned + centre) * QUARTERS).astype(np.int16)
    _, first = np.unique(placed.reshape(len(placed), -1), axis=0, return_index=True)
    first.sort()
    stride = round(step * QUARTERS)
    quarters, parameters = [], []
    for pattern, turn in zip(placed[first], turns[first], strict=True):
        low = -(pattern.min(axis=0) // stride)
        high = (LAST - pattern.max(axis=0)) // stride
        rows, columns = np.meshgrid(
            np.arange(low[0], high[0] + 1),
            np.arange(low[1], high[1] + 1),
            indexing="ij",
        )
        moves = stride * np.stack((rows.ravel(), columns.ravel()), axis=-1)
        quarters.append(pattern + moves[:, None].astype(np.int16))
        parameters.append(
            np.c_[np.broadcast_to(turn, (len(moves), 2)), moves / QUARTERS]
        )
    return ImagePoses(np.concatenate(quarters), np.concatenate(parameters))


def window(wavelengths: np.ndarray) -> np.ndarray:
    return np.round((wavelengths - SPECTRUM_ORIGIN) / SPECTRUM_TOKEN_WIDTH, DECIMALS)


def spectrum_poses(slots: np.ndarray) -> SpectrumPoses:
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
    return SpectrumPoses(
        np.minimum(reads, N_SPECTRUM_TOKENS - 1), weights.astype(np.float32), stretches
    )
