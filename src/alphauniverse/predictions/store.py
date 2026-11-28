from functools import cache

import numpy as np
import pyarrow as pa
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage

from alphauniverse.config import (
    HSC_TABLE_COLUMNS,
    KEPT,
    LS_TABLE_COLUMNS,
    REDSHIFT,
    SPECTRUM_MODES,
    SPECTRUM_TOKEN_RANK,
    TABLE_MODES,
    TOP_CODES,
    artifact,
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


def row(galaxy: int) -> pa.RecordBatch:
    return predictions().slice(galaxy, 1).to_batches()[0]


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


def dense_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    codes, probabilities, spread = query_image_tokens(rows, survey, slots)
    dense = np.repeat(
        spread[..., None], MODALITY_INFO["tok_image"]["vocab_size"], axis=-1
    )
    np.put_along_axis(dense, codes, probabilities, axis=-1)
    return dense


def spectrum_token_coefficients(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    quantized = array(
        rows, f"{survey}_coefficients", DESISpectrum.num_tokens - 1, SPECTRUM_TOKEN_RANK
    )[:, slots]
    offsets = array(rows, f"{survey}_offsets", DESISpectrum.num_tokens - 1)[:, slots]
    steps = array(rows, f"{survey}_steps", DESISpectrum.num_tokens - 1)[:, slots]
    return offsets[..., None] + quantized * steps[..., None]


def table_value_probabilities(rows: pa.RecordBatch, mode: str) -> np.ndarray:
    column, count, vocabulary = TABLE_MODES[mode]
    return array(rows, column, count, vocabulary)


def distributions(mode: str, form: np.ndarray) -> np.ndarray:
    if mode in SPECTRUM_MODES:
        mean, directions, _, _ = basis()[SPECTRUM_MODES[mode]]
        return mean + form @ directions
    return form


@cache
def aion_predictions(mode: str) -> np.ndarray:
    return np.load(artifact(f"aion_prediction_{mode}"), mmap_mode="r")


def table_value_aion_predictions(galaxy: int) -> dict[str, float]:
    found = {
        column: float(value)
        for survey, columns in (("ls", LS_TABLE_COLUMNS), ("hsc", HSC_TABLE_COLUMNS))
        for column, value in zip(
            columns, aion_predictions(f"{survey}_table")[galaxy], strict=True
        )
    }
    found[REDSHIFT] = float(aion_predictions(REDSHIFT)[galaxy, 0])
    return found
