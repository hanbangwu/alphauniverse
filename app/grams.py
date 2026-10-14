from collections.abc import Callable
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


class Forms(NamedTuple):
    dense: Callable[[slice], torch.Tensor]
    shift: float
    floor: float
    count: int


def padded(vocabulary: int) -> int:
    return vocabulary + -vocabulary % ALIGNMENT


def stacked(
    table: pa.Table, parts: Callable[[pa.RecordBatch], tuple[np.ndarray, ...]]
) -> list[torch.Tensor]:
    stored, start = [], 0
    for rows in table.to_batches():
        tensors = [torch.tensor(array) for array in parts(rows)]
        if not stored:
            stored = [
                tensor.new_empty((table.num_rows, *tensor.shape[1:]), device=device())
                for tensor in tensors
            ]
        for target, tensor in zip(stored, tensors, strict=True):
            target[start : start + rows.num_rows] = tensor
        start += rows.num_rows
    return stored


def image_forms(survey: str, table: pa.Table) -> Forms:
    def parts(rows: pa.RecordBatch) -> tuple[np.ndarray, ...]:
        codes, probabilities, spread = pql.query_image_tokens(rows, survey, slice(None))
        return codes.astype(np.int64), probabilities, spread

    codes, probabilities, spread = stacked(table, parts)

    def dense(part: slice) -> torch.Tensor:
        tile = spread.new_zeros((*spread[part].shape, padded(IMAGE_VOCABULARY)))
        tile[..., :IMAGE_VOCABULARY] = spread[part, :, None]
        return tile.scatter_(-1, codes[part], probabilities[part])

    return Forms(dense, 0.0, PROBABILITY_FLOOR, table.num_rows)


def spectrum_forms(survey: str, table: pa.Table) -> Forms:
    _, _, mean_square, projected_mean = pql.basis()[survey]
    (coefficients,) = stacked(
        table,
        lambda rows: (
            pql.spectrum_token_coefficients(rows, survey, slice(None)) + projected_mean,
        ),
    )
    return Forms(
        lambda part: coefficients[part],
        float(mean_square - projected_mean @ projected_mean),
        pql.SPECTRUM_TOKEN_FLOOR,
        table.num_rows,
    )


def table_forms(mode: str, table: pa.Table) -> Forms:
    def parts(rows: pa.RecordBatch) -> tuple[np.ndarray]:
        values = pql.table_value_probabilities(rows, mode)
        padding = padded(values.shape[-1]) - values.shape[-1]
        return (np.pad(values, ((0, 0), (0, 0), (0, padding))),)

    (values,) = stacked(table, parts)
    return Forms(lambda part: values[part], 0.0, PROBABILITY_FLOOR, table.num_rows)


def forms(mode: str, table: pa.Table) -> Forms:
    if mode in IMAGE_MODES:
        return image_forms(IMAGE_MODES[mode], table)
    if mode in SPECTRUM_MODES:
        return spectrum_forms(SPECTRUM_MODES[mode], table)
    return table_forms(mode, table)


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


def gram(
    mode: str, queries: Forms, gallery: Forms, symmetric: bool
) -> dict[str, torch.Tensor]:
    size = ROWS // pql.WIDTHS[mode]
    found = {
        name: torch.empty(queries.count, gallery.count, device=device())
        for name in TOKEN_GRAMS[mode]
    }
    for first in tqdm(range(0, queries.count, size), desc=mode):
        rows = slice(first, first + size)
        left = queries.dense(rows)
        for second in range(first if symmetric else 0, gallery.count, size):
            columns = slice(second, second + size)
            right = left if symmetric and second == first else gallery.dense(columns)
            same, forward, backward = scores(left, right, gallery)
            for matrix, direct, transposed in zip(
                found.values(), (same, forward), (same.T, backward.T)
            ):
                matrix[rows, columns] = direct
                if symmetric and second > first:
                    matrix[columns, rows] = transposed
    return found


def generate_token_grams() -> None:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    for mode in TOKEN_GRAMS:
        everyone = forms(mode, pql.predictions())
        for name, matrix in gram(mode, everyone, everyone, symmetric=True).items():
            np.save(artifact(f"token_gram_{name}"), matrix.cpu().numpy())
