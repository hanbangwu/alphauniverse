from functools import cache

import numpy as np
import pyarrow as pa

from .config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_PATCHES,
    N_SPANS,
    PREDICTIONS,
    SCALAR_SURVEYS,
    SPAN_RANK,
    SPECTRUM_SURVEYS,
    TOP_CODES,
    VOCABULARY,
    artifact,
)
from .search import N_LS_SCALARS, Query, tokens

KEPT = 16
SPAN_FLOOR = 0.1 / VOCABULARY


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
    scalars = np.asarray(query.scalars, dtype=np.int64)
    selected = {
        f"{ANCHOR}_cells": np.asarray(query.patches, dtype=np.int64),
        f"{ANCHOR}_scalars": scalars[scalars < N_LS_SCALARS],
        "hsc_scalars": scalars[scalars >= N_LS_SCALARS] - N_LS_SCALARS,
    }
    if query.spans:
        survey = spectrum_survey(query.galaxy)
        selected[f"{survey}_spans"] = np.asarray(query.spans, dtype=np.int64)
    return {mode: slots for mode, slots in selected.items() if len(slots)}


def top_cells(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray, kept: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = array(rows, f"{survey}_codes", N_PATCHES, TOP_CODES)[:, slots, :kept]
    probabilities = np.exp(
        array(rows, f"{survey}_log_probabilities", N_PATCHES, TOP_CODES)[:, slots],
        dtype=np.float32,
    )
    rest = np.exp(array(rows, f"{survey}_tails", N_PATCHES)[:, slots]) + (
        probabilities[..., kept:].sum(axis=-1)
    )
    return codes, probabilities[..., :kept], rest / (IMAGE_VOCABULARY - kept)


def dense_cells(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    codes, probabilities, spread = top_cells(rows, survey, slots, TOP_CODES)
    dense = np.repeat(spread[..., None], IMAGE_VOCABULARY, axis=-1)
    np.put_along_axis(dense, codes.astype(np.intp), probabilities, axis=-1)
    return dense


def cell_overlaps(
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


def span_coefficients(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    quantised = array(rows, f"{survey}_coefficients", N_SPANS, SPAN_RANK)[:, slots]
    offsets = array(rows, f"{survey}_offsets", N_SPANS)[:, slots]
    steps = array(rows, f"{survey}_steps", N_SPANS)[:, slots]
    return offsets[..., None] + quantised * steps[..., None]


def span_overlaps(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray, query: np.ndarray
) -> np.ndarray:
    _, _, mean_square, projected_mean = basis()[survey]
    weights = projected_mean + query
    quantised = array(rows, f"{survey}_coefficients", N_SPANS, SPAN_RANK)[:, slots]
    overlaps = (
        mean_square
        + query @ projected_mean
        + array(rows, f"{survey}_offsets", N_SPANS)[:, slots] * weights.sum(axis=-1)
        + array(rows, f"{survey}_steps", N_SPANS)[:, slots]
        * np.einsum("...r,...r->...", quantised, weights)
    )
    return np.maximum(overlaps, SPAN_FLOOR)


def scalar_probabilities(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    count = len(SCALAR_SURVEYS[survey])
    stored = array(rows, f"{survey}_scalars", count, VOCABULARY)[:, slots]
    return np.exp(stored.astype(np.float32))


def query_forms(
    own: pa.RecordBatch, selected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    forms = {}
    for mode, slots in selected.items():
        survey, kind = mode.split("_")
        if kind == "cells":
            forms[mode] = dense_cells(own, survey, slots)[0]
        elif kind == "spans":
            forms[mode] = span_coefficients(own, survey, slots)[0]
        else:
            forms[mode] = scalar_probabilities(own, survey, slots)[0]
    return forms


def overlaps(
    rows: pa.RecordBatch, mode: str, slots: np.ndarray, form: np.ndarray
) -> np.ndarray:
    survey, kind = mode.split("_")
    if kind == "cells":
        return cell_overlaps(form, *top_cells(rows, survey, slots, KEPT))
    if kind == "spans":
        return span_overlaps(rows, survey, slots, form)
    return (scalar_probabilities(rows, survey, slots) * form).sum(axis=-1)


def sums(
    rows: pa.RecordBatch,
    selected: dict[str, np.ndarray],
    forms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        mode: np.log(overlaps(rows, mode, slots, forms[mode])).sum(axis=-1)
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
    cells = np.full((len(galaxies), N_PATCHES), np.nan)
    spans = np.full((len(galaxies), N_SPANS), np.nan)
    for mode, form in forms.items():
        survey, kind = mode.split("_")
        if kind == "cells":
            cells = np.log(
                cell_overlaps(
                    form.mean(axis=0), *top_cells(rows, survey, slice(None), KEPT)
                )
            )
        elif kind == "spans":
            spans = np.log(span_overlaps(rows, survey, slice(None), form.mean(axis=0)))
    scalars = np.hstack(
        [
            np.log(
                (
                    scalar_probabilities(rows, survey, slice(None))
                    * scalar_probabilities(own, survey, slice(None))
                ).sum(axis=-1)
            )
            for survey in SCALAR_SURVEYS
        ]
    )
    return cells, spans, scalars
