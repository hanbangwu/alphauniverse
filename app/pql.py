from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from functools import cache, partial
from itertools import accumulate
from threading import Lock
from typing import Annotated, NamedTuple, Self, get_args

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
    TokenMode,
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
BLOCK_BYTES = 64 * 2**20
SALIENCY_MATCHES = 256
SALIENCY_BACKGROUND = 256
BUILDING = Lock()
MATRICES = OrderedDict()
SUMS = OrderedDict()

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
    galaxy: int
    values: np.ndarray
    peaks: np.ndarray
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
def basis() -> dict[str, tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]]:
    with np.load(artifact("prediction_basis")) as stored:
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


def cross_overlaps(
    rows: pa.RecordBatch, mode: str, form: np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    if mode in IMAGE_MODES:
        codes, probabilities, spread = top_image_tokens(
            rows, IMAGE_MODES[mode], slice(None), KEPT
        )
        codes = codes.astype(np.intp)
        folded = probabilities - spread[..., None]
        block = max(1, BLOCK_BYTES // (4 * codes.size))
        for start in range(0, len(form), block):
            part = form[start : start + block]
            yield (
                start,
                np.einsum("gck,bgck->bgc", folded, part[:, codes])
                + spread * part.sum(axis=-1)[:, None, None],
            )
        return
    survey = SPECTRUM_MODES[mode]
    _, _, mean_square, projected_mean = basis()[survey]
    quantised = array(
        rows, f"{survey}_coefficients", N_SPECTRUM_TOKENS, SPECTRUM_TOKEN_RANK
    ).astype(np.float32)
    offsets = array(rows, f"{survey}_offsets", N_SPECTRUM_TOKENS)
    steps = array(rows, f"{survey}_steps", N_SPECTRUM_TOKENS)
    block = max(1, BLOCK_BYTES // (4 * offsets.size))
    for start in range(0, len(form), block):
        part = form[start : start + block].astype(np.float32)
        weights = projected_mean + part
        overlaps = (
            mean_square
            + part @ projected_mean
            + offsets[..., None] * weights.sum(axis=-1)
            + steps[..., None] * (quantised @ weights.T)
        )
        yield start, np.maximum(overlaps, SPECTRUM_TOKEN_FLOOR).transpose(2, 0, 1)


def gathered(galaxies: np.ndarray) -> pa.RecordBatch:
    rows = [predictions().slice(galaxy, 1) for galaxy in galaxies.tolist()]
    return pa.concat_tables(rows).combine_chunks().to_batches()[0]


def best_log_overlaps(rows: pa.RecordBatch, mode: str, form: np.ndarray) -> np.ndarray:
    best = np.empty((rows.num_rows, len(form)))
    for start, overlaps in cross_overlaps(rows, mode, form):
        best[:, start : start + len(overlaps)] = overlaps.max(axis=-1).T
    return np.log(best)


def basket_maps(
    galaxy: int,
    galaxies: np.ndarray,
    selected: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shown, positions = {}, {}
    if not selected:
        return shown, positions
    rows = gathered(galaxies)
    for mode, form in query_forms(row(galaxy), selected).items():
        scale = np.exp(-peaks[mode])
        shown[mode] = np.zeros((len(galaxies), WIDTHS[mode]), np.float32)
        positions[mode] = np.empty((len(galaxies), len(form)), np.int32)
        for start, overlaps in cross_overlaps(rows, mode, form):
            stop = start + len(overlaps)
            shown[mode] = np.maximum(
                shown[mode], (overlaps * scale[start:stop, None, None]).max(axis=0)
            )
            positions[mode][:, start:stop] = overlaps.argmax(axis=-1).T
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
    return np.concatenate(
        [
            best_log_overlaps(rows, mode, form).sum(axis=1)
            for rows in predictions().to_batches()
        ]
    )


def distributions(mode: str, form: np.ndarray) -> np.ndarray:
    if mode in SPECTRUM_MODES:
        mean, directions, _, _ = basis()[SPECTRUM_MODES[mode]]
        return mean + form @ directions
    return form


def log_peaks(mode: str, form: np.ndarray) -> np.ndarray:
    if mode in TABLE_MODES:
        return form.max(axis=-1)
    return np.log(distributions(mode, form).max(axis=-1))


def sums(
    rows: pa.RecordBatch,
    selected: dict[str, np.ndarray],
    forms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        mode: log_overlaps(rows, mode, slots, forms[mode]).sum(axis=-1)
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


def built(galaxy: int) -> Agreements:
    forms = query_forms(row(galaxy), dict.fromkeys(MODES, slice(None)))
    values = np.empty((predictions().num_rows, WIDTH), np.float32)
    start = 0
    for rows in predictions().to_batches():
        stop = start + rows.num_rows
        for mode, form in forms.items():
            values[start:stop, OFFSETS[mode] : OFFSETS[mode] + WIDTHS[mode]] = (
                log_overlaps(rows, mode, slice(None), form)
            )
        start = stop
    others = np.arange(len(values)) != galaxy
    means = (values.sum(axis=0, dtype=np.float64) - values[galaxy]) / others.sum()
    squares = np.zeros(WIDTH)
    for start in range(0, len(values), BATCH):
        block = values[start : start + BATCH].astype(np.float64) - means
        squares += (block[others[start : start + BATCH]] ** 2).sum(axis=0)
    return Agreements(
        galaxy,
        values,
        np.concatenate([log_peaks(mode, forms[mode]) for mode in MODES]),
        np.sqrt(squares / others.sum()),
        values.sum(axis=1, dtype=np.float64),
    )


def products(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.zeros((values.shape[1], weights.shape[1]))
    for start in range(0, len(values), BATCH):
        stop = start + BATCH
        total += values[start:stop].astype(np.float64).T @ weights[start:stop]
    return total


def column_sums(
    found: Agreements, selected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        mode: found.values[:, OFFSETS[mode] + slots].sum(axis=1, dtype=np.float64)
        for mode, slots in selected.items()
    }


def selected_peaks(
    found: Agreements, selected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        mode: found.peaks[OFFSETS[mode] + slots] for mode, slots in selected.items()
    }


def mode_totals(
    found: Agreements,
    selected: dict[str, np.ndarray],
    positional: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aligned = {
        mode: slots for mode, slots in selected.items() if mode not in positional
    }
    return column_sums(found, aligned) | basket_sums(found.galaxy, positional)


def positional_slots(
    selected: dict[str, np.ndarray], anywhere: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {mode: selected[mode] for mode in anywhere}


def scores(query: Selection) -> np.ndarray:
    selected = selection(query)
    return combine(
        mode_totals(
            agreements(query.galaxy),
            selected,
            positional_slots(selected, query.anywhere),
        )
    )


def columns(selected: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([OFFSETS[mode] + slots for mode, slots in selected.items()])


def by_mode(values: np.ndarray) -> dict[str, np.ndarray]:
    table = OFFSETS[next(iter(TABLE_MODES))]
    return {
        mode: values[..., OFFSETS[mode] : OFFSETS[mode] + WIDTHS[mode]]
        for mode in (*IMAGE_MODES, *SPECTRUM_MODES)
    } | {"table_values": values[..., table:]}


def maps(found: Agreements, galaxies: np.ndarray) -> dict[str, np.ndarray]:
    return by_mode(np.exp(found.values[galaxies] - found.peaks))


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


def saliency_sample(query: Selection) -> np.ndarray:
    order = np.argsort(-scores(query), kind="stable")
    order = order[order != query.galaxy]
    rest = order[SALIENCY_MATCHES:]
    background = np.random.default_rng(query.galaxy).choice(
        rest, min(SALIENCY_BACKGROUND, len(rest)), replace=False
    )
    return np.concatenate([order[:SALIENCY_MATCHES], background])


def sampled_values(query: Selection, found: Agreements) -> np.ndarray:
    sample = saliency_sample(query)
    values = found.values[sample].astype(np.float64)
    rows = gathered(sample)
    forms = query_forms(row(query.galaxy), dict.fromkeys(query.anywhere, slice(None)))
    for mode, form in forms.items():
        values[:, OFFSETS[mode] : OFFSETS[mode] + WIDTHS[mode]] = best_log_overlaps(
            rows, mode, form
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
    found = agreements(query.galaxy)
    totals = mode_totals(found, selected, positional)
    scored = combine(totals)
    shares = fractions(selected_peaks(found, selected), totals)
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    shown, positions = basket_maps(
        query.galaxy, galaxies, positional, selected_peaks(found, positional)
    )
    return Results(
        galaxies,
        scored[galaxies],
        similarity(shares)[galaxies],
        {mode: values[galaxies] for mode, values in totals.items()},
        {mode: np.exp(values[galaxies]) for mode, values in shares.items()},
        predicted(selected, galaxies),
        maps(found, galaxies) | shown,
        positions,
    )
