from functools import cache
from typing import Annotated, NamedTuple, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    N_TABLE_VALUES,
    OBSERVATIONS,
    PREDICTIONS,
    REDSHIFT,
    SPECTRUM_MODES,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_RANK,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    TOP_CODES,
    VOCABULARY,
    GalaxyIndex,
    artifact,
)
from .search import tokens

KEPT = 16
SPECTRUM_TOKEN_FLOOR = 0.1 / VOCABULARY
OBSERVED_IN = {
    mode: survey
    for mode, survey in (
        IMAGE_MODES
        | SPECTRUM_MODES
        | {f"{survey}_table": survey for survey in TABLE_VALUE_SURVEYS}
        | {REDSHIFT: REDSHIFT}
    ).items()
    if survey in OBSERVATIONS
}
TABLE_SLOTS = [
    (mode, slot) for mode, (_, count, _) in TABLE_MODES.items() for slot in range(count)
]

ImageTokens = Annotated[
    tuple[Annotated[int, Field(ge=0, lt=N_IMAGE_TOKENS)], ...],
    Field(max_length=N_IMAGE_TOKENS),
]
SpectrumTokens = Annotated[
    tuple[Annotated[int, Field(ge=0, lt=N_SPECTRUM_TOKENS)], ...],
    Field(max_length=N_SPECTRUM_TOKENS),
]


@cache
def observed(column: str) -> np.ndarray:
    return pc.is_valid(tokens().column(column)).to_numpy()


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    galaxy: GalaxyIndex
    ls_image: ImageTokens = ()
    hsc_image: ImageTokens = ()
    desi_spectrum: SpectrumTokens = ()
    sdss_spectrum: SpectrumTokens = ()
    table_values: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_TABLE_VALUES)], ...],
        Field(max_length=N_TABLE_VALUES),
    ] = ()
    matches: Annotated[int, Field(ge=1, le=128)] = 32

    @model_validator(mode="after")
    def selects_observed_slots(self) -> Self:
        selected = selection(self)
        if not selected:
            raise ValueError(
                "select at least one image token, spectrum token or table value"
            )
        for mode in selected.keys() & OBSERVED_IN.keys():
            column = OBSERVED_IN[mode]
            if not observed(column)[self.galaxy]:
                raise ValueError(f"galaxy {self.galaxy} has no {OBSERVATIONS[column]}")
        return self


class Results(NamedTuple):
    galaxies: np.ndarray
    scores: np.ndarray
    sums: dict[str, np.ndarray]
    aligned: dict[str, np.ndarray]
    selected: dict[str, np.ndarray]


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
    rows = [predictions().slice(galaxy, 1) for galaxy in galaxies.tolist()]
    return pa.concat_tables(rows).combine_chunks().to_batches()[0]


def row(galaxy: int) -> pa.RecordBatch:
    return predictions().slice(galaxy, 1).to_batches()[0]


def selection(query: Query) -> dict[str, np.ndarray]:
    selected = {mode: getattr(query, mode) for mode in (*IMAGE_MODES, *SPECTRUM_MODES)}
    for table_value in query.table_values:
        mode, slot = TABLE_SLOTS[table_value]
        selected[mode] = (*selected.get(mode, ()), slot)
    return {mode: np.unique(slots) for mode, slots in selected.items() if len(slots)}


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
    rows: pa.RecordBatch, mode: str, slots: slice | np.ndarray
) -> np.ndarray:
    column, count, vocabulary = TABLE_MODES[mode]
    return array(rows, column, count, vocabulary)[:, slots].astype(np.float32)


def table_value_overlaps(gallery: np.ndarray, query: np.ndarray) -> np.ndarray:
    products = gallery + query
    largest = products.max(axis=-1)
    return largest + np.log(np.exp(products - largest[..., None]).sum(axis=-1))


def query_forms(
    own: pa.RecordBatch, selected: dict[str, np.ndarray | slice]
) -> dict[str, np.ndarray]:
    forms = {}
    for mode, slots in selected.items():
        if mode in IMAGE_MODES:
            forms[mode] = dense_image_tokens(own, IMAGE_MODES[mode], slots)[0]
        elif mode in SPECTRUM_MODES:
            forms[mode] = spectrum_token_coefficients(own, SPECTRUM_MODES[mode], slots)[
                0
            ]
        else:
            forms[mode] = table_value_log_probabilities(own, mode, slots)[0]
    return forms


def log_overlaps(
    rows: pa.RecordBatch, mode: str, slots: np.ndarray | slice, form: np.ndarray
) -> np.ndarray:
    if mode in IMAGE_MODES:
        return np.log(
            image_token_overlaps(
                form, *top_image_tokens(rows, IMAGE_MODES[mode], slots, KEPT)
            )
        )
    if mode in SPECTRUM_MODES:
        return np.log(spectrum_token_overlaps(rows, SPECTRUM_MODES[mode], slots, form))
    return table_value_overlaps(table_value_log_probabilities(rows, mode, slots), form)


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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    own = row(query.galaxy)
    rows = gathered(galaxies)
    every = query_forms(
        own, dict.fromkeys((*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES), slice(None))
    )
    aligned = {
        mode: log_overlaps(rows, mode, slice(None), form)
        for mode, form in every.items()
    }
    aligned["table_values"] = np.hstack([aligned.pop(mode) for mode in TABLE_MODES])
    selected = {
        mode: log_overlaps(rows, mode, slice(None), every[mode][slots].mean(axis=0))
        for mode, slots in selection(query).items()
        if mode not in TABLE_MODES
    }
    return aligned, selected


def search(query: Query) -> Results:
    totals = parts(query)
    scored = combine(totals)
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return Results(
        galaxies,
        scored[galaxies],
        {mode: values[galaxies] for mode, values in totals.items()},
        *maps(query, galaxies),
    )
