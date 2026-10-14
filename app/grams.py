from collections.abc import Callable
from time import perf_counter
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from . import pql
from .config import (
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    PROBABILITY_FLOOR,
    SPECTRUM_MODES,
    TOKEN_GRAMS,
    artifact,
    device,
)

ALIGNMENT = 32
ROWS = 64 * N_IMAGE_TOKENS

ImageTokens = Callable[
    [pa.RecordBatch, str, slice], tuple[np.ndarray, np.ndarray, np.ndarray]
]


class Forms(NamedTuple):
    rows: Callable[[slice], torch.Tensor]
    shift: float
    floor: float


def blocks(mode: str) -> tuple[str, ...]:
    return tuple(name for name in (mode, f"{mode}_anywhere") if name in TOKEN_GRAMS)


def width(vocabulary: int) -> int:
    return vocabulary + -vocabulary % ALIGNMENT


def stacked(
    parts: Callable[[pa.RecordBatch], tuple[np.ndarray, ...]],
) -> list[torch.Tensor]:
    count = pql.predictions().num_rows
    stored, start = [], 0
    for rows in pql.predictions().to_batches():
        tensors = [torch.tensor(array) for array in parts(rows)]
        if not stored:
            stored = [
                tensor.new_empty((count, *tensor.shape[1:]), device=device())
                for tensor in tensors
            ]
        for target, tensor in zip(stored, tensors, strict=True):
            target[start : start + rows.num_rows] = tensor
        start += rows.num_rows
    return stored


def image_forms(survey: str, image_tokens: ImageTokens) -> Forms:
    def parts(rows: pa.RecordBatch) -> tuple[np.ndarray, ...]:
        codes, probabilities, spread = image_tokens(rows, survey, slice(None))
        return codes.astype(np.int64), probabilities, spread

    codes, probabilities, spread = stacked(parts)

    def rows(part: slice) -> torch.Tensor:
        dense = spread.new_zeros((*spread[part].shape, width(IMAGE_VOCABULARY)))
        dense[..., :IMAGE_VOCABULARY] = spread[part, :, None]
        return dense.scatter_(-1, codes[part], probabilities[part])

    return Forms(rows, 0.0, PROBABILITY_FLOOR)


def spectrum_forms(survey: str) -> Forms:
    _, _, mean_square, projected_mean = pql.basis()[survey]
    (coefficients,) = stacked(
        lambda rows: (
            pql.spectrum_token_coefficients(rows, survey, slice(None)) + projected_mean,
        )
    )
    return Forms(
        lambda part: coefficients[part],
        float(mean_square - projected_mean @ projected_mean),
        pql.SPECTRUM_TOKEN_FLOOR,
    )


def table_forms(mode: str) -> Forms:
    def parts(rows: pa.RecordBatch) -> tuple[np.ndarray]:
        values = pql.table_value_probabilities(rows, mode)
        padding = width(values.shape[-1]) - values.shape[-1]
        return (np.pad(values, ((0, 0), (0, 0), (0, padding))),)

    (values,) = stacked(parts)
    return Forms(lambda part: values[part], 0.0, PROBABILITY_FLOOR)


def forms(mode: str, image_tokens: ImageTokens = pql.query_image_tokens) -> Forms:
    if mode in IMAGE_MODES:
        return image_forms(IMAGE_MODES[mode], image_tokens)
    if mode in SPECTRUM_MODES:
        return spectrum_forms(SPECTRUM_MODES[mode])
    return table_forms(mode)


def summed(overlaps: torch.Tensor, forms: Forms, dim: int) -> torch.Tensor:
    return (
        (overlaps + forms.shift)
        .clamp_min_(forms.floor)
        .log_()
        .sum(dim, dtype=torch.float64)
    )


def scores(
    left: torch.Tensor, right: torch.Tensor, forms: Forms
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count, slots, _ = left.shape
    products = (left.flatten(0, 1) @ right.flatten(0, 1).T).view(
        count, slots, len(right), slots
    )
    return (
        summed(products.diagonal(dim1=1, dim2=3), forms, -1),
        summed(products.amax(dim=3), forms, 1),
        summed(products.amax(dim=1), forms, 2),
    )


def gram(mode: str, forms: Forms) -> dict[str, torch.Tensor]:
    count = pql.predictions().num_rows
    size = max(1, ROWS // pql.WIDTHS[mode])
    names = blocks(mode)
    found = {name: torch.empty(count, count, device=device()) for name in names}
    for first in tqdm(range(0, count, size), desc=mode):
        rows = slice(first, first + size)
        left = forms.rows(rows)
        for second in range(first, count, size):
            columns = slice(second, second + size)
            right = left if second == first else forms.rows(columns)
            same, forward, backward = scores(left, right, forms)
            for name, direct, transposed in zip(
                names, (same, forward), (same.T, backward.T)
            ):
                found[name][rows, columns] = direct
                if second > first:
                    found[name][columns, rows] = transposed
    return found


def generate_token_grams() -> None:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    for mode in pql.MODES:
        start = perf_counter()
        stored = forms(mode)
        loaded = perf_counter()
        found = {
            name: block.cpu().numpy() for name, block in gram(mode, stored).items()
        }
        print(
            f"{mode}: forms {loaded - start:.1f} s, "
            f"products {perf_counter() - loaded:.1f} s"
        )
        for name, block in found.items():
            np.save(artifact(f"token_gram_{name}"), block)
