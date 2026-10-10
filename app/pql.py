from functools import cache

import numpy as np
import pyarrow as pa

from .config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_PATCHES,
    N_SPANS,
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
        f"{spectrum_survey(query.galaxy)}_spans": np.asarray(
            query.spans, dtype=np.int64
        ),
        f"{ANCHOR}_scalars": scalars[scalars < N_LS_SCALARS],
        "hsc_scalars": scalars[scalars >= N_LS_SCALARS] - N_LS_SCALARS,
    }
    return {mode: slots for mode, slots in selected.items() if len(slots)}


def dense_cells(rows: pa.RecordBatch, survey: str) -> np.ndarray:
    codes = array(rows, f"{survey}_codes", N_PATCHES, TOP_CODES).astype(np.int64)
    kept = np.exp(
        array(rows, f"{survey}_log_probabilities", N_PATCHES, TOP_CODES).astype(
            np.float32
        )
    )
    rest = np.exp(array(rows, f"{survey}_tails", N_PATCHES)) / (
        IMAGE_VOCABULARY - TOP_CODES
    )
    dense = np.repeat(rest[..., None], IMAGE_VOCABULARY, axis=-1)
    np.put_along_axis(dense, codes, kept, axis=-1)
    return dense


def top_cells(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = array(rows, f"{survey}_codes", N_PATCHES, TOP_CODES)[:, slots, :KEPT]
    probabilities = np.exp(
        array(rows, f"{survey}_log_probabilities", N_PATCHES, TOP_CODES)[
            :, slots
        ].astype(np.float32)
    )
    rest = np.exp(array(rows, f"{survey}_tails", N_PATCHES)[:, slots]) + (
        probabilities[..., KEPT:].sum(axis=-1)
    )
    return (
        codes.astype(np.int64),
        probabilities[..., :KEPT],
        rest / (IMAGE_VOCABULARY - KEPT),
    )


def cell_overlaps(
    query: np.ndarray,
    codes: np.ndarray,
    probabilities: np.ndarray,
    spread: np.ndarray,
) -> np.ndarray:
    at_codes = np.take_along_axis(
        np.broadcast_to(query, (*codes.shape[:-1], query.shape[-1])),
        codes,
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


def span_overlaps(query: np.ndarray, gallery: np.ndarray, survey: str) -> np.ndarray:
    _, _, mean_square, projected_mean = basis()[survey]
    overlaps = (
        mean_square
        + query @ projected_mean
        + gallery @ projected_mean
        + (gallery * query).sum(axis=-1)
    )
    return np.maximum(overlaps, SPAN_FLOOR)


def scalar_probabilities(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    count = len(SCALAR_SURVEYS[survey])
    stored = array(rows, f"{survey}_scalars", count, VOCABULARY)[:, slots]
    return np.exp(stored.astype(np.float32))


def query_forms(query: Query, selected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    row = gathered(np.asarray([query.galaxy]))
    forms = {}
    for mode, slots in selected.items():
        survey, kind = mode.split("_")
        if kind == "cells":
            forms[mode] = dense_cells(row, survey)[0, slots]
        elif kind == "spans":
            forms[mode] = span_coefficients(row, survey, slots)[0]
        else:
            forms[mode] = scalar_probabilities(row, survey, slots)[0]
    return forms


def overlaps(
    rows: pa.RecordBatch, mode: str, slots: np.ndarray, form: np.ndarray
) -> np.ndarray:
    survey, kind = mode.split("_")
    if kind == "cells":
        return cell_overlaps(form, *top_cells(rows, survey, slots))
    if kind == "spans":
        return span_overlaps(form, span_coefficients(rows, survey, slots), survey)
    return (scalar_probabilities(rows, survey, slots) * form).sum(axis=-1)


def parts(query: Query) -> dict[str, np.ndarray]:
    selected = selection(query)
    forms = query_forms(query, selected)
    sums = {mode: np.empty(predictions().num_rows) for mode in selected}
    start = 0
    for rows in predictions().to_batches():
        stop = start + rows.num_rows
        for mode, slots in selected.items():
            sums[mode][start:stop] = np.log(
                overlaps(rows, mode, slots, forms[mode])
            ).sum(axis=-1)
        start = stop
    return sums


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
    selected = selection(query)
    forms = query_forms(query, selected)
    rows = gathered(galaxies)
    cells = np.full((len(galaxies), N_PATCHES), np.nan)
    spans = np.full((len(galaxies), N_SPANS), np.nan)
    for mode, form in forms.items():
        survey, kind = mode.split("_")
        if kind == "cells":
            cells = np.log(
                cell_overlaps(form.mean(axis=0), *top_cells(rows, survey, slice(None)))
            )
        elif kind == "spans":
            spans = np.log(
                span_overlaps(
                    form.mean(axis=0),
                    span_coefficients(rows, survey, slice(None)),
                    survey,
                )
            )
    own = gathered(np.asarray([query.galaxy]))
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
