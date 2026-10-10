from functools import cache

import numpy as np
import pyarrow as pa

from .config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    PREDICTIONS,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_RANK,
    TABLE_VALUE_SURVEYS,
    TOP_CODES,
    VOCABULARY,
    artifact,
)
from .search import FIRST_HSC_TABLE_VALUE, FIRST_LS_TABLE_VALUE, Query, tokens

KEPT = 16
SPECTRUM_TOKEN_FLOOR = 0.1 / VOCABULARY


def prediction_batch(records: list[dict[str, np.ndarray]]) -> pa.RecordBatch:
    columns = []
    for field in PREDICTIONS:
        flat = np.concatenate([values[field.name] for values in records])
        columns.append(
            pa.FixedSizeListArray.from_arrays(flat, field.type.list_size)
            if pa.types.is_fixed_size_list(field.type)
            else pa.array(flat)
        )
    return pa.record_batch(columns, schema=PREDICTIONS)


def save_prediction_basis(fitted: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    np.savez(
        artifact("prediction_basis"),
        **{
            f"{survey}_{name}": value
            for survey, pair in fitted.items()
            for name, value in zip(("mean", "directions"), pair, strict=True)
        },
    )


@cache
def predictions() -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(artifact("predictions")))).read_all()


@cache
def basis() -> dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    with np.load(artifact("prediction_basis")) as stored:
        fitted = {}
        for survey in SPECTRUM_SURVEYS:
            mean = stored[f"{survey}_mean"]
            directions = stored[f"{survey}_directions"]
            fitted[survey] = (mean, directions, mean @ mean, directions @ mean)
        return fitted


def array(rows: pa.RecordBatch, name: str, *shape: int) -> np.ndarray:
    return rows.column(name).flatten().to_numpy().reshape(rows.num_rows, *shape)


def gathered(galaxies: np.ndarray) -> pa.RecordBatch:
    return predictions().take(galaxies).combine_chunks().to_batches()[0]


def row(galaxy: int) -> pa.RecordBatch:
    return predictions().slice(galaxy, 1).to_batches()[0]


def spectrum_survey(galaxy: int) -> str | None:
    return next(
        (
            survey
            for survey in SPECTRUM_SURVEYS
            if tokens().column(survey)[galaxy].is_valid
        ),
        None,
    )


def selection(query: Query) -> dict[str, np.ndarray]:
    table_values = np.asarray(query.table_values, dtype=np.int64)
    selected = {
        f"{ANCHOR}_image": np.asarray(query.image_tokens, dtype=np.int64),
        f"{ANCHOR}_table": table_values[
            (table_values >= FIRST_LS_TABLE_VALUE)
            & (table_values < FIRST_HSC_TABLE_VALUE)
        ]
        - FIRST_LS_TABLE_VALUE,
        "hsc_table": table_values[table_values >= FIRST_HSC_TABLE_VALUE]
        - FIRST_HSC_TABLE_VALUE,
    }
    if query.spectrum_tokens:
        survey = spectrum_survey(query.galaxy)
        selected[f"{survey}_spectrum"] = np.asarray(
            query.spectrum_tokens, dtype=np.int64
        )
    return {mode: slots for mode, slots in selected.items() if len(slots)}


def top_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray, kept: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = array(rows, f"{survey}_codes", N_IMAGE_TOKENS, TOP_CODES)[:, slots, :kept]
    probabilities = np.exp(
        array(rows, f"{survey}_log_probabilities", N_IMAGE_TOKENS, TOP_CODES)[:, slots],
        dtype=np.float32,
    )
    rest = np.exp(array(rows, f"{survey}_tails", N_IMAGE_TOKENS)[:, slots]) + (
        probabilities[..., kept:].sum(axis=-1)
    )
    return codes, probabilities[..., :kept], rest / (IMAGE_VOCABULARY - kept)


def dense_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    codes, probabilities, spread = top_image_tokens(rows, survey, slots, TOP_CODES)
    dense = np.repeat(spread[..., None], IMAGE_VOCABULARY, axis=-1)
    np.put_along_axis(dense, codes.astype(np.intp), probabilities, axis=-1)
    return dense


def image_token_overlaps(
    query: np.ndarray,
    codes: np.ndarray,
    probabilities: np.ndarray,
    spread: np.ndarray,
) -> np.ndarray:
    at_codes = np.take_along_axis(
        np.broadcast_to(query, (*codes.shape[:-1], query.shape[-1])),
        codes.astype(np.intp),
        axis=-1,
    )
    return (probabilities * at_codes).sum(axis=-1) + spread * (
        query.sum(axis=-1) - at_codes.sum(axis=-1)
    )


def spectrum_token_coefficients(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    quantised = array(
        rows, f"{survey}_coefficients", N_SPECTRUM_TOKENS, SPECTRUM_TOKEN_RANK
    )[:, slots]
    offsets = array(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS)[:, slots]
    steps = array(rows, f"{survey}_steps", N_SPECTRUM_TOKENS)[:, slots]
    return offsets[..., None] + quantised * steps[..., None]


def spectrum_token_overlaps(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray, query: np.ndarray
) -> np.ndarray:
    _, _, mean_square, projected_mean = basis()[survey]
    weights = projected_mean + query
    quantised = array(
        rows, f"{survey}_coefficients", N_SPECTRUM_TOKENS, SPECTRUM_TOKEN_RANK
    )[:, slots]
    overlaps = (
        mean_square
        + query @ projected_mean
        + array(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS)[:, slots]
        * weights.sum(axis=-1)
        + array(rows, f"{survey}_steps", N_SPECTRUM_TOKENS)[:, slots]
        * np.einsum("...r,...r->...", quantised, weights)
    )
    return np.maximum(overlaps, SPECTRUM_TOKEN_FLOOR)


def table_value_log_probabilities(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    count = len(TABLE_VALUE_SURVEYS[survey])
    stored = array(rows, f"{survey}_table_values", count, VOCABULARY)[:, slots]
    return stored.astype(np.float32)


def table_value_overlaps(gallery: np.ndarray, query: np.ndarray) -> np.ndarray:
    products = gallery + query
    largest = products.max(axis=-1)
    return largest + np.log(np.exp(products - largest[..., None]).sum(axis=-1))


def query_forms(
    own: pa.RecordBatch, selected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    forms = {}
    for mode, slots in selected.items():
        survey, kind = mode.split("_")
        if kind == "image":
            forms[mode] = dense_image_tokens(own, survey, slots)[0]
        elif kind == "spectrum":
            forms[mode] = spectrum_token_coefficients(own, survey, slots)[0]
        else:
            forms[mode] = table_value_log_probabilities(own, survey, slots)[0]
    return forms


def log_overlaps(
    rows: pa.RecordBatch, mode: str, slots: np.ndarray, form: np.ndarray
) -> np.ndarray:
    survey, kind = mode.split("_")
    if kind == "image":
        return np.log(
            image_token_overlaps(form, *top_image_tokens(rows, survey, slots, KEPT))
        )
    if kind == "spectrum":
        return np.log(spectrum_token_overlaps(rows, survey, slots, form))
    return table_value_overlaps(
        table_value_log_probabilities(rows, survey, slots), form
    )


def sums(
    rows: pa.RecordBatch,
    selected: dict[str, np.ndarray],
    forms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        mode: log_overlaps(rows, mode, slots, forms[mode]).sum(axis=-1)
        for mode, slots in selected.items()
    }


def parts(query: Query) -> dict[str, np.ndarray]:
    selected = selection(query)
    forms = query_forms(row(query.galaxy), selected)
    totals = {mode: np.empty(predictions().num_rows) for mode in selected}
    start = 0
    for rows in predictions().to_batches():
        stop = start + rows.num_rows
        for mode, values in sums(rows, selected, forms).items():
            totals[mode][start:stop] = values
        start = stop
    return totals


def combine(sums: dict[str, np.ndarray]) -> np.ndarray:
    if len(sums) == 1:
        return next(iter(sums.values()))
    return np.mean(
        [(values - values.mean()) / values.std() for values in sums.values()],
        axis=0,
    )


def scores(query: Query) -> np.ndarray:
    return combine(parts(query))


def maps(
    query: Query, galaxies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    own = row(query.galaxy)
    forms = query_forms(own, selection(query))
    rows = gathered(galaxies)
    image_tokens = np.full((len(galaxies), N_IMAGE_TOKENS), np.nan)
    spectrum_tokens = np.full((len(galaxies), N_SPECTRUM_TOKENS), np.nan)
    for mode, form in forms.items():
        survey, kind = mode.split("_")
        if kind == "image":
            image_tokens = np.log(
                image_token_overlaps(
                    form.mean(axis=0),
                    *top_image_tokens(rows, survey, slice(None), KEPT),
                )
            )
        elif kind == "spectrum":
            spectrum_tokens = np.log(
                spectrum_token_overlaps(rows, survey, slice(None), form.mean(axis=0))
            )
    table_values = np.hstack(
        [
            table_value_overlaps(
                table_value_log_probabilities(rows, survey, slice(None)),
                table_value_log_probabilities(own, survey, slice(None)),
            )
            for survey in TABLE_VALUE_SURVEYS
        ]
    )
    return image_tokens, spectrum_tokens, table_values
