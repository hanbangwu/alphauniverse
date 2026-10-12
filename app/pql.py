import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from itertools import accumulate
from threading import Lock
from typing import Annotated, NamedTuple, Self, get_args

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    KEPT,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    N_TABLE_VALUES,
    OBSERVATIONS,
    PREDICTIONS,
    PROBABILITY_FLOOR,
    REDSHIFT,
    SPECTRUM_MODES,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_RANK,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    TOP_CODES,
    VOCABULARY,
    GalaxyIndex,
    TokenMode,
    artifact,
)
from .search import tokens

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
QUERY_COLUMNS = [
    f"{survey}_{name}"
    for survey in IMAGE_MODES.values()
    for name in ("query_codes", "query_probabilities", "tails")
]
TABLE_SLOTS = [
    (mode, slot) for mode, (_, count, _) in TABLE_MODES.items() for slot in range(count)
]

WIDTHS = (
    dict.fromkeys(IMAGE_MODES, N_IMAGE_TOKENS)
    | dict.fromkeys(SPECTRUM_MODES, N_SPECTRUM_TOKENS)
    | {mode: count for mode, (_, count, _) in TABLE_MODES.items()}
)
MODES = tuple(WIDTHS)
OFFSETS = {
    mode: end - WIDTHS[mode]
    for mode, end in zip(MODES, accumulate(WIDTHS.values()), strict=True)
}
WIDTH = sum(WIDTHS.values())
AGREEMENT_GALAXIES = 8
BATCH = 256
CHUNK = 16
BLOCK = 8192
SALIENCY_SAMPLE = 512
BUILDING = Lock()
MATRICES = OrderedDict()
SUMS = OrderedDict()
THREADS = pa.cpu_count()

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


class Selection(BaseModel):
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
    anywhere: Annotated[
        tuple[TokenMode, ...], Field(max_length=len(get_args(TokenMode)))
    ] = ()

    @model_validator(mode="after")
    def selects_observed_slots(self) -> Self:
        selected = selection(self)
        if not selected:
            raise ValueError(
                "select at least one image token, spectrum token or table value"
            )
        if missing := set(self.anywhere) - selected.keys():
            raise ValueError(
                f"no selection in position-independent {', '.join(sorted(missing))}"
            )
        for mode in selected.keys() & OBSERVED_IN.keys():
            column = OBSERVED_IN[mode]
            if not observed(column)[self.galaxy]:
                raise ValueError(f"galaxy {self.galaxy} has no {OBSERVATIONS[column]}")
        return self


class Query(Selection):
    matches: Annotated[int, Field(ge=1, le=128)] = 32


class Results(NamedTuple):
    galaxies: np.ndarray
    scores: np.ndarray
    similarity: np.ndarray
    sums: dict[str, np.ndarray]
    similarities: dict[str, np.ndarray]
    predicted: list[list[str]]
    maps: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]


class Agreements(NamedTuple):
    values: np.ndarray
    spread: np.ndarray
    totals: np.ndarray


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
        artifact("probability_basis"),
        **{
            f"{survey}_{name}": value
            for survey, pair in fitted.items()
            for name, value in zip(("mean", "directions"), pair, strict=True)
        },
    )


@cache
def predictions() -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(artifact("probabilities")))).read_all()


@cache
def basis() -> dict[str, tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]]:
    with np.load(artifact("probability_basis")) as stored:
        fitted = {}
        for survey in SPECTRUM_SURVEYS:
            mean = stored[f"{survey}_mean"].astype(np.float32)
            directions = stored[f"{survey}_directions"].astype(np.float32)
            fitted[survey] = (mean, directions, mean @ mean, directions @ mean)
        return fitted


def array(rows: pa.RecordBatch, name: str, *shape: int) -> np.ndarray:
    return rows.column(name).flatten().to_numpy().reshape(rows.num_rows, *shape)


def row(galaxy: int) -> pa.RecordBatch:
    return predictions().slice(galaxy, 1).to_batches()[0]


def selection(query: Selection) -> dict[str, np.ndarray]:
    selected = {mode: getattr(query, mode) for mode in (*IMAGE_MODES, *SPECTRUM_MODES)}
    for table_value in query.table_values:
        mode, slot = TABLE_SLOTS[table_value]
        selected[mode] = (*selected.get(mode, ()), slot)
    return {mode: np.unique(slots) for mode, slots in selected.items() if len(slots)}


def top_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = array(rows, f"{survey}_codes", N_IMAGE_TOKENS, KEPT)[:, slots]
    probabilities = array(rows, f"{survey}_probabilities", N_IMAGE_TOKENS, KEPT)[
        :, slots
    ]
    rest = array(rows, f"{survey}_rest", N_IMAGE_TOKENS)[:, slots]
    return codes, probabilities, rest / (IMAGE_VOCABULARY - KEPT)


def dense_image_tokens(
    rows: pa.RecordBatch, survey: str, slots: slice | np.ndarray
) -> np.ndarray:
    codes, probabilities, _ = top_image_tokens(rows, survey, slots)
    width = TOP_CODES - KEPT
    query_codes = array(rows, f"{survey}_query_codes", N_IMAGE_TOKENS, width)[:, slots]
    query_probabilities = array(
        rows, f"{survey}_query_probabilities", N_IMAGE_TOKENS, width
    )[:, slots]
    tails = array(rows, f"{survey}_tails", N_IMAGE_TOKENS)[:, slots]
    dense = np.repeat(
        (tails / (IMAGE_VOCABULARY - TOP_CODES))[..., None], IMAGE_VOCABULARY, axis=-1
    )
    np.put_along_axis(dense, codes, probabilities, axis=-1)
    np.put_along_axis(dense, query_codes, query_probabilities, axis=-1)
    return dense


def image_token_overlaps(
    query: np.ndarray,
    codes: np.ndarray,
    probabilities: np.ndarray,
    spread: np.ndarray,
) -> np.ndarray:
    flat = query.ravel()
    offsets = np.arange(len(query))[:, None] * query.shape[-1]
    totals = query.sum(axis=-1)
    overlaps = np.empty(codes.shape[:-1], np.float32)
    step = max(1, BLOCK // len(codes))
    for start in range(0, len(query), step):
        block = slice(start, start + step)
        weights = probabilities[:, block] - spread[:, block, None]
        at_codes = np.take(flat, codes[:, block] + offsets[block])
        overlaps[:, block] = (
            np.einsum("gsk,gsk->gs", weights, at_codes)
            + spread[:, block] * totals[block]
        )
    return overlaps


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
        + np.einsum("...r,r->...", query, projected_mean)
        + array(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS)[:, slots]
        * weights.sum(axis=-1)
        + array(rows, f"{survey}_steps", N_SPECTRUM_TOKENS)[:, slots]
        * np.einsum("...r,...r->...", quantised, weights)
    )
    return np.maximum(overlaps, SPECTRUM_TOKEN_FLOOR)


def table_value_probabilities(rows: pa.RecordBatch, mode: str) -> np.ndarray:
    column, count, vocabulary = TABLE_MODES[mode]
    return array(rows, column, count, vocabulary)


def table_value_overlaps(
    gallery: np.ndarray, slots: slice | np.ndarray, query: np.ndarray
) -> np.ndarray:
    overlaps = np.empty((len(gallery), len(query)), np.float32)
    for index, slot in enumerate(np.arange(gallery.shape[1])[slots]):
        overlaps[:, index] = gallery[:, slot] @ query[index]
    return overlaps


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
            forms[mode] = table_value_probabilities(own, mode)[0, slots]
    return forms


def galaxy_forms(galaxy: int) -> dict[str, np.ndarray]:
    return query_forms(row(galaxy), dict.fromkeys(MODES, slice(None)))


def log_overlaps(
    rows: pa.RecordBatch, mode: str, slots: np.ndarray | slice, form: np.ndarray
) -> np.ndarray:
    if mode in IMAGE_MODES:
        overlaps = image_token_overlaps(
            form, *top_image_tokens(rows, IMAGE_MODES[mode], slots)
        )
    elif mode in SPECTRUM_MODES:
        overlaps = spectrum_token_overlaps(rows, SPECTRUM_MODES[mode], slots, form)
    else:
        overlaps = table_value_overlaps(
            table_value_probabilities(rows, mode), slots, form
        )
    return np.log(np.maximum(overlaps, PROBABILITY_FLOOR, out=overlaps))


def cross_overlaps(
    rows: pa.RecordBatch, mode: str, form: np.ndarray
) -> Iterator[tuple[slice, np.ndarray]]:
    if mode in IMAGE_MODES:
        codes, probabilities, spread = top_image_tokens(
            rows, IMAGE_MODES[mode], slice(None)
        )
        indices = torch.from_numpy(codes.astype(np.int64))
        weights = torch.from_numpy(probabilities - spread[..., None])
        spread = torch.from_numpy(spread)
        table = torch.from_numpy(np.ascontiguousarray(form.T))
        totals = torch.from_numpy(form.sum(axis=-1))
        for start in range(0, rows.num_rows, CHUNK):
            chunk = slice(start, start + CHUNK)
            overlaps = torch.nn.functional.embedding_bag(
                indices[chunk].flatten(0, 1),
                table,
                per_sample_weights=weights[chunk].flatten(0, 1),
                mode="sum",
            ).view(-1, N_IMAGE_TOKENS, len(form))
            yield chunk, overlaps.addcmul_(spread[chunk, :, None], totals).numpy()
        return
    survey = SPECTRUM_MODES[mode]
    _, _, mean_square, projected_mean = basis()[survey]
    quantised = array(
        rows, f"{survey}_coefficients", N_SPECTRUM_TOKENS, SPECTRUM_TOKEN_RANK
    )
    offsets = array(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS)[..., None]
    steps = array(rows, f"{survey}_steps", N_SPECTRUM_TOKENS)[..., None]
    weights = projected_mean + form
    base = mean_square + np.einsum("br,r->b", form, projected_mean)
    totals = weights.sum(axis=-1)
    for start in range(0, rows.num_rows, CHUNK):
        chunk = slice(start, start + CHUNK)
        overlaps = quantised[chunk].astype(np.float32) @ weights.T
        overlaps *= steps[chunk]
        overlaps += offsets[chunk] * totals
        overlaps += base
        yield chunk, np.maximum(overlaps, SPECTRUM_TOKEN_FLOOR, out=overlaps)


def threaded(
    compute: Callable[[pa.RecordBatch], np.ndarray], batches: list[pa.RecordBatch]
) -> np.ndarray:
    with ThreadPoolExecutor(THREADS) as pool:
        return np.concatenate(list(pool.map(compute, batches)))


def slices(rows: pa.RecordBatch) -> list[pa.RecordBatch]:
    size = math.ceil(rows.num_rows / THREADS)
    return [rows.slice(start, size) for start in range(0, rows.num_rows, size)]


def gathered(galaxies: np.ndarray) -> pa.RecordBatch:
    gallery = predictions().drop_columns(QUERY_COLUMNS)
    rows = [gallery.slice(galaxy, 1) for galaxy in galaxies.tolist()]
    return pa.concat_tables(rows).combine_chunks().to_batches()[0]


def best_log_overlaps(rows: pa.RecordBatch, mode: str, form: np.ndarray) -> np.ndarray:
    best = np.empty((rows.num_rows, len(form)))
    for chunk, overlaps in cross_overlaps(rows, mode, form):
        best[chunk] = overlaps.max(axis=1)
    return np.log(np.maximum(best, PROBABILITY_FLOOR, out=best))


def basket_maps(
    galaxy: int,
    rows: pa.RecordBatch,
    selected: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shown, positions = {}, {}
    for mode, form in query_forms(row(galaxy), selected).items():
        scale = np.exp(-peaks[mode])
        shown[mode] = np.empty((rows.num_rows, WIDTHS[mode]), np.float32)
        positions[mode] = np.empty((rows.num_rows, len(form)), np.int32)
        for chunk, overlaps in cross_overlaps(rows, mode, form):
            shown[mode][chunk] = (overlaps * scale).max(axis=-1)
            positions[mode][chunk] = overlaps.argmax(axis=1)
    return shown, positions


def basket_sums(galaxy: int, selected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        mode: cached(
            SUMS,
            (galaxy, mode, *slots.tolist()),
            partial(basket_sum, galaxy, mode, slots),
        )
        for mode, slots in selected.items()
    }


def cached[T](store: OrderedDict, key: Hashable, compute: Callable[[], T]) -> T:
    if (found := store.get(key)) is not None:
        return found
    with BUILDING:
        if key not in store:
            store[key] = compute()
            if len(store) > AGREEMENT_GALAXIES:
                store.popitem(last=False)
        return store[key]


def basket_sum(galaxy: int, mode: str, slots: np.ndarray) -> np.ndarray:
    form = query_forms(row(galaxy), {mode: slots})[mode]
    return threaded(
        lambda rows: best_log_overlaps(rows, mode, form).sum(axis=1),
        predictions().to_batches(),
    )


def distributions(mode: str, form: np.ndarray) -> np.ndarray:
    if mode in SPECTRUM_MODES:
        mean, directions, _, _ = basis()[SPECTRUM_MODES[mode]]
        return mean + form @ directions
    return form


def slot_peaks(forms: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.log(distributions(mode, forms[mode]).max(axis=-1)) for mode in MODES]
    )


def sums(
    rows: pa.RecordBatch,
    selected: dict[str, np.ndarray],
    forms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        mode: log_overlaps(rows, mode, slots, forms[mode]).sum(
            axis=-1, dtype=np.float64
        )
        for mode, slots in selected.items()
    }


def selected_forms(
    query: Query,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected = selection(query)
    return selected, query_forms(row(query.galaxy), selected)


def combine(totals: dict[str, np.ndarray]) -> np.ndarray:
    if len(totals) == 1:
        return next(iter(totals.values()))
    return np.mean(
        [(values - values.mean()) / values.std() for values in totals.values()],
        axis=0,
    )


def cosines(galaxy: int, mode: str) -> np.ndarray:
    form = distributions(mode, query_forms(row(galaxy), {mode: slice(None)})[mode])
    unit = form / np.linalg.norm(form, axis=-1, keepdims=True)
    return (unit @ unit.T).astype(np.float32, copy=False)


def agreements(galaxy: int) -> Agreements:
    return cached(MATRICES, galaxy, partial(built, galaxy))


def agreement_rows(rows: pa.RecordBatch, forms: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [log_overlaps(rows, mode, slice(None), forms[mode]) for mode in MODES], axis=1
    )


def built(galaxy: int) -> Agreements:
    values = threaded(
        partial(agreement_rows, forms=galaxy_forms(galaxy)), predictions().to_batches()
    )
    others = np.arange(len(values)) != galaxy
    means = (values.sum(axis=0, dtype=np.float64) - values[galaxy]) / others.sum()
    squares = np.zeros(WIDTH)
    for start in range(0, len(values), BATCH):
        block = values[start : start + BATCH].astype(np.float64) - means
        squares += (block[others[start : start + BATCH]] ** 2).sum(axis=0)
    return Agreements(
        values, np.sqrt(squares / others.sum()), values.sum(axis=1, dtype=np.float64)
    )


def products(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.zeros((values.shape[1], weights.shape[1]))
    for start in range(0, len(values), BATCH):
        stop = start + BATCH
        total += values[start:stop].astype(np.float64).T @ weights[start:stop]
    return total


def column_sums(galaxy: int, selected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    forms = query_forms(row(galaxy), selected)
    batches = [sums(rows, selected, forms) for rows in predictions().to_batches()]
    return {
        mode: np.concatenate([batch[mode] for batch in batches]) for mode in selected
    }


def selected_peaks(
    peaks: np.ndarray, selected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {mode: peaks[OFFSETS[mode] + slots] for mode, slots in selected.items()}


def mode_totals(
    galaxy: int,
    selected: dict[str, np.ndarray],
    positional: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aligned = {
        mode: slots for mode, slots in selected.items() if mode not in positional
    }
    return column_sums(galaxy, aligned) | basket_sums(galaxy, positional)


def positional_slots(
    selected: dict[str, np.ndarray], anywhere: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {mode: selected[mode] for mode in anywhere}


def scores(query: Selection) -> np.ndarray:
    selected = selection(query)
    return combine(
        mode_totals(query.galaxy, selected, positional_slots(selected, query.anywhere))
    )


def columns(selected: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([OFFSETS[mode] + slots for mode, slots in selected.items()])


def by_mode(values: np.ndarray) -> dict[str, np.ndarray]:
    table = OFFSETS[next(iter(TABLE_MODES))]
    return {
        mode: values[..., OFFSETS[mode] : OFFSETS[mode] + WIDTHS[mode]]
        for mode in (*IMAGE_MODES, *SPECTRUM_MODES)
    } | {"table_values": values[..., table:]}


def maps(
    forms: dict[str, np.ndarray], peaks: np.ndarray, rows: pa.RecordBatch
) -> dict[str, np.ndarray]:
    return by_mode(np.exp(agreement_rows(rows, forms) - peaks))


def saliency(query: Selection) -> np.ndarray:
    found = agreements(query.galaxy)
    chosen = columns(selection(query))
    if not query.anywhere:
        others = np.arange(len(found.values)) != query.galaxy
        return partial_correlations(
            found.values, found.spread, found.totals, chosen, others
        )
    values = sampled_values(query, found)
    return partial_correlations(
        values,
        values.std(axis=0),
        values.sum(axis=1),
        chosen,
        np.ones(len(values), bool),
    )


def saliency_sample(galaxy: int) -> np.ndarray:
    others = np.delete(np.arange(predictions().num_rows), galaxy)
    return np.random.default_rng(galaxy).choice(
        others, min(SALIENCY_SAMPLE, len(others)), replace=False
    )


def sampled_values(query: Selection, found: Agreements) -> np.ndarray:
    sample = saliency_sample(query.galaxy)
    values = found.values[sample].astype(np.float64)
    parts = slices(gathered(sample))
    forms = query_forms(row(query.galaxy), dict.fromkeys(query.anywhere, slice(None)))
    for mode, form in forms.items():
        values[:, OFFSETS[mode] : OFFSETS[mode] + WIDTHS[mode]] = threaded(
            partial(best_log_overlaps, mode=mode, form=form), parts
        )
    return values


def partial_correlations(
    values: np.ndarray,
    spread: np.ndarray,
    totals: np.ndarray,
    chosen: np.ndarray,
    others: np.ndarray,
) -> np.ndarray:
    count = others.sum()
    score = values[:, chosen].sum(axis=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        control = (totals - score) / (WIDTH - len(chosen))
        targets = np.c_[score, control]
        targets = np.where(others[:, None], targets - targets[others].mean(axis=0), 0)
        spreads = np.sqrt((targets**2).sum(axis=0) / count)
        covariances = products(values, targets).T
        with_score, with_control = covariances / (count * spread * spreads[:, None])
        between = targets[:, 0] @ targets[:, 1] / (count * spreads.prod())
        partial = (with_score - with_control * between) / np.sqrt(
            (1 - with_control**2) * (1 - between**2)
        )
    return np.clip(np.nan_to_num(partial, nan=0), -1, 1).astype(np.float32)


def fractions(
    peaks: dict[str, np.ndarray], totals: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        mode: (values - peaks[mode].sum()) / len(peaks[mode])
        for mode, values in totals.items()
    }


def similarity(shares: dict[str, np.ndarray]) -> np.ndarray:
    if len(shares) == 1:
        return np.exp(next(iter(shares.values())))
    weights = {mode: 1 / values.std() for mode, values in shares.items()}
    weighted = sum(weights[mode] * values for mode, values in shares.items())
    return np.exp(weighted / sum(weights.values()))


def predicted(selected: dict[str, np.ndarray], galaxies: np.ndarray) -> list[list[str]]:
    needed = {OBSERVED_IN[mode] for mode in selected.keys() & OBSERVED_IN.keys()}
    return [
        [
            column
            for column in OBSERVATIONS
            if column in needed and not observed(column)[galaxy]
        ]
        for galaxy in galaxies.tolist()
    ]


def search(query: Query) -> Results:
    selected = selection(query)
    positional = positional_slots(selected, query.anywhere)
    forms = galaxy_forms(query.galaxy)
    peaks = slot_peaks(forms)
    totals = mode_totals(query.galaxy, selected, positional)
    scored = combine(totals)
    shares = fractions(selected_peaks(peaks, selected), totals)
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    rows = gathered(galaxies)
    shown, positions = basket_maps(
        query.galaxy, rows, positional, selected_peaks(peaks, positional)
    )
    return Results(
        galaxies,
        scored[galaxies],
        similarity(shares)[galaxies],
        {mode: values[galaxies] for mode, values in totals.items()},
        {mode: np.exp(values[galaxies]) for mode, values in shares.items()},
        predicted(selected, galaxies),
        maps(forms, peaks, rows) | shown,
        positions,
    )
