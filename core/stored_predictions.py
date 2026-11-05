from functools import cache

import numpy as np
import pyarrow as pa
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage

from .artifacts import artifact
from .config import (
    IMAGE_MODES,
    KEPT,
    SPECTRUM_MODES,
    SPECTRUM_TOKEN_RANK,
    TABLE_MODES,
    TOP_CODES,
)

SPECTRUM_TOKEN_FLOOR = 0.1 / MODALITY_INFO["tok_spectrum_desi"]["vocab_size"]

WIDTHS = (
    dict.fromkeys(IMAGE_MODES, LegacySurveyImage.num_tokens)
    | dict.fromkeys(SPECTRUM_MODES, DESISpectrum.num_tokens - 1)
    | {mode: count for mode, (_, count, _) in TABLE_MODES.items()}
)


@cache
def predictions() -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(artifact("probabilities")))).read_all()


@cache
def prediction_basis() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(artifact("probability_basis")) as stored:
        return {
            survey: (stored[f"{survey}_mean"], stored[f"{survey}_directions"])
            for survey in SPECTRUM_MODES.values()
        }


@cache
def basis() -> dict[str, tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]]:
    fitted = {}
    for survey, pair in prediction_basis().items():
        mean, directions = (part.astype(np.float32) for part in pair)
        fitted[survey] = (mean, directions, mean @ mean, directions @ mean)
    return fitted


def array(rows: pa.RecordBatch, name: str, *shape: int) -> np.ndarray:
    return rows.column(name).flatten().to_numpy().reshape(rows.num_rows, *shape)


def top_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = array(rows, f"{survey}_codes", LegacySurveyImage.num_tokens, KEPT)[:, slots]
    probabilities = array(
        rows, f"{survey}_probabilities", LegacySurveyImage.num_tokens, KEPT
    )[:, slots]
    rest = array(rows, f"{survey}_rest", LegacySurveyImage.num_tokens)[:, slots]
    return (
        codes,
        probabilities,
        rest / (MODALITY_INFO["tok_image"]["vocab_size"] - KEPT),
    )


def query_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, probabilities, _ = top_image_tokens(rows, survey, slots)
    width = TOP_CODES - KEPT
    query_codes = array(
        rows, f"{survey}_query_codes", LegacySurveyImage.num_tokens, width
    )[:, slots]
    query_probabilities = array(
        rows, f"{survey}_query_probabilities", LegacySurveyImage.num_tokens, width
    )[:, slots]
    tails = array(rows, f"{survey}_tails", LegacySurveyImage.num_tokens)[:, slots]
    return (
        np.concatenate((codes, query_codes), axis=-1),
        np.concatenate((probabilities, query_probabilities), axis=-1),
        tails / (MODALITY_INFO["tok_image"]["vocab_size"] - TOP_CODES),
    )


def spectrum_token_coefficients(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    quantised = array(
        rows, f"{survey}_coefficients", DESISpectrum.num_tokens - 1, SPECTRUM_TOKEN_RANK
    )[:, slots]
    offsets = array(rows, f"{survey}_offsets", DESISpectrum.num_tokens - 1)[:, slots]
    steps = array(rows, f"{survey}_steps", DESISpectrum.num_tokens - 1)[:, slots]
    return offsets[..., None] + quantised * steps[..., None]


def table_value_probabilities(rows: pa.RecordBatch, mode: str) -> np.ndarray:
    column, count, vocabulary = TABLE_MODES[mode]
    return array(rows, column, count, vocabulary)


def distributions(mode: str, form: np.ndarray) -> np.ndarray:
    if mode in SPECTRUM_MODES:
        mean, directions, _, _ = basis()[SPECTRUM_MODES[mode]]
        return mean + form @ directions
    return form
