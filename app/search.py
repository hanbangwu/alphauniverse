from functools import cache
from typing import Annotated, Self

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow.fs import LocalFileSystem
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    ANCHOR,
    ARTIFACTS,
    DIM,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    N_TABLE_VALUES,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    SEED,
    SPECTRUM_SURVEYS,
    TABLE_VALUE_SURVEYS,
    GalaxyIndex,
    artifact,
)

BATCH = 256
NLIST = 16384
NPROBE = 64
PROBE = 2048
TRAIN_GALAXIES = 2048
N_LS_TABLE_VALUES = len(TABLE_VALUE_SURVEYS[ANCHOR])
N_HSC_TABLE_VALUES = len(TABLE_VALUE_SURVEYS["hsc"])
FIRST_LS_TABLE_VALUE = REDSHIFT_TABLE_VALUE + 1
FIRST_HSC_TABLE_VALUE = FIRST_LS_TABLE_VALUE + N_LS_TABLE_VALUES


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    galaxy: GalaxyIndex
    image_tokens: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_IMAGE_TOKENS)], ...],
        Field(alias="p", max_length=N_IMAGE_TOKENS),
    ] = ()
    spectrum_tokens: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_SPECTRUM_TOKENS)], ...],
        Field(alias="s", max_length=N_SPECTRUM_TOKENS),
    ] = ()
    table_values: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_TABLE_VALUES)], ...],
        Field(alias="t", max_length=N_TABLE_VALUES),
    ] = ()
    matches: Annotated[int, Field(ge=1, le=128)] = 32

    @model_validator(mode="after")
    def some_tokens(self) -> Self:
        if not (self.image_tokens or self.spectrum_tokens or self.table_values):
            raise ValueError(
                "select at least one image token, spectrum token or table value"
            )
        return self

    @model_validator(mode="after")
    def spectrum_tokens_need_a_spectrum(self) -> Self:
        if self.spectrum_tokens and not with_spectrum()[self.galaxy]:
            raise ValueError(f"galaxy {self.galaxy} has no spectrum")
        return self

    @model_validator(mode="after")
    def hsc_table_values_need_hsc(self) -> Self:
        hsc = any(
            table_value >= FIRST_HSC_TABLE_VALUE for table_value in self.table_values
        )
        if hsc and not with_hsc()[self.galaxy]:
            raise ValueError(f"galaxy {self.galaxy} has no HSC match")
        return self

    @model_validator(mode="after")
    def redshift_needs_a_redshift(self) -> Self:
        if (
            REDSHIFT_TABLE_VALUE in self.table_values
            and not with_redshift()[self.galaxy]
        ):
            raise ValueError(f"galaxy {self.galaxy} has no usable redshift")
        return self


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(
        artifact(role),
        format=ARTIFACTS[role],
        filesystem=LocalFileSystem(use_mmap=True),
    )


@cache
def tokens() -> pa.Table:
    return source("tokens").to_table()


@cache
def with_hsc() -> np.ndarray:
    return pc.is_valid(tokens().column("hsc")).to_numpy()


@cache
def with_spectrum() -> np.ndarray:
    return pc.is_valid(spectrum_cells(tokens())).to_numpy()


@cache
def with_redshift() -> np.ndarray:
    return pc.is_valid(tokens().column(REDSHIFT)).to_numpy()


def layout(
    has_spectrum: np.ndarray, has_hsc: np.ndarray, has_redshift: np.ndarray
) -> np.ndarray:
    sizes = (
        N_IMAGE_TOKENS
        + N_LS_TABLE_VALUES
        + has_spectrum * N_SPECTRUM_TOKENS
        + has_hsc * N_HSC_TABLE_VALUES
        + has_redshift
    )
    return np.concatenate(([0], np.cumsum(sizes)))


def table_value_starts(first: np.ndarray, has_spectrum: np.ndarray) -> np.ndarray:
    return first + N_IMAGE_TOKENS + has_spectrum * N_SPECTRUM_TOKENS


@cache
def bounds() -> np.ndarray:
    return layout(with_spectrum(), with_hsc(), with_redshift())


@cache
def starts() -> np.ndarray:
    return bounds()[:-1]


def positions(first: np.ndarray, width: int) -> np.ndarray:
    return (first[:, None] + np.arange(width)).reshape(-1)


def centroid(query: Query, *, index: faiss.Index) -> np.ndarray:
    start = starts()[query.galaxy]
    table_value_start = table_value_starts(start, with_spectrum()[query.galaxy])
    table_values = np.asarray(query.table_values, dtype=np.int64)
    ids = np.concatenate(
        (
            start + np.asarray(query.image_tokens, dtype=np.int64),
            start + N_IMAGE_TOKENS + np.asarray(query.spectrum_tokens, dtype=np.int64),
            np.where(
                table_values == REDSHIFT_TABLE_VALUE,
                bounds()[query.galaxy + 1] - 1,
                table_value_start + table_values - FIRST_LS_TABLE_VALUE,
            ),
        )
    )
    direction = index.reconstruct_batch(ids).mean(axis=0, keepdims=True)
    faiss.normalize_L2(direction)
    return direction


def table_value_tokens(galaxy: int) -> dict[str, int]:
    found = {
        column: int(token)
        for survey, columns in TABLE_VALUE_SURVEYS.items()
        if (cell := tokens().column(survey)[galaxy]).is_valid
        for column, token in zip(
            columns, np.asarray(cell.values)[N_IMAGE_TOKENS:], strict=True
        )
    }
    if (cell := tokens().column(REDSHIFT)[galaxy]).is_valid:
        found[REDSHIFT] = int(cell.values[0].as_py())
    return found


def candidates(
    query: Query, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    nearest, lists = PROBE, NPROBE
    while True:
        ids = index.search(
            direction, nearest, params=faiss.SearchParametersIVF(nprobe=lists)
        )[1][0]
        found, first = np.unique(
            np.searchsorted(starts(), ids[ids >= 0], side="right") - 1,
            return_index=True,
        )
        ranked = found[np.argsort(first)]
        others = ranked[ranked != query.galaxy][: query.matches]
        if len(others) == query.matches or (
            lists >= index.nlist and nearest >= index.ntotal
        ):
            return np.concatenate(([query.galaxy], others)).astype(np.int32)
        nearest, lists = 2 * nearest, 2 * lists


def vectors(order: np.ndarray, *, index: faiss.Index) -> np.ndarray:
    return index.reconstruct_batch(positions(starts()[order], N_IMAGE_TOKENS))


def score_maps(rows: np.ndarray, direction: np.ndarray, *, width: int) -> np.ndarray:
    return (rows @ direction.T).reshape(-1, width)


def optional_maps(
    first: np.ndarray,
    has: np.ndarray,
    direction: np.ndarray,
    *,
    width: int,
    index: faiss.Index,
) -> np.ndarray:
    maps = np.full((len(first), width), np.nan, dtype=np.float32)
    rows = index.reconstruct_batch(positions(first[has], width))
    maps[has] = score_maps(rows, direction, width=width)
    return maps


def spectrum_token_maps(
    order: np.ndarray, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    return optional_maps(
        starts()[order] + N_IMAGE_TOKENS,
        with_spectrum()[order],
        direction,
        width=N_SPECTRUM_TOKENS,
        index=index,
    )


def table_value_maps(
    order: np.ndarray, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    has_spectrum, has_hsc = with_spectrum()[order], with_hsc()[order]
    first = table_value_starts(starts()[order], has_spectrum)
    rows = index.reconstruct_batch(positions(first, N_LS_TABLE_VALUES))
    return np.hstack(
        (
            optional_maps(
                bounds()[order + 1] - 1,
                with_redshift()[order],
                direction,
                width=1,
                index=index,
            ),
            score_maps(rows, direction, width=N_LS_TABLE_VALUES),
            optional_maps(
                first + N_LS_TABLE_VALUES,
                has_hsc,
                direction,
                width=N_HSC_TABLE_VALUES,
                index=index,
            ),
        )
    )


def rank(order: np.ndarray, *maps: np.ndarray) -> tuple[np.ndarray, ...]:
    scores = np.fmax.reduce([np.fmax.reduce(scored, axis=1) for scored in maps])
    by_score = np.concatenate(([0], 1 + np.argsort(-scores[1:], kind="stable")))
    return order[by_score], scores[by_score], *(scored[by_score] for scored in maps)


def search(
    query: Query, *, index: faiss.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = centroid(query, index=index)
    order = candidates(query, direction, index=index)
    return rank(
        order,
        score_maps(vectors(order, index=index), direction, width=N_IMAGE_TOKENS),
        spectrum_token_maps(order, direction, index=index),
        table_value_maps(order, direction, index=index),
    )


def rows(cells: pa.Array | pa.ChunkedArray, start: int, stop: int | None) -> np.ndarray:
    flat = pc.list_flatten(pc.list_flatten(pc.list_slice(cells, start, stop)))
    embeddings = np.asarray(flat, dtype=np.float32).reshape(-1, DIM)
    if not np.isfinite(embeddings.sum()):
        raise ValueError("embeddings must be finite")
    faiss.normalize_L2(embeddings)
    return embeddings


def image_tokens(cells: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return rows(cells, 0, N_IMAGE_TOKENS)


def spectral(cells: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return rows(cells, 1, None)


def spectrum_cells(batch: pa.RecordBatch | pa.Table) -> pa.Array | pa.ChunkedArray:
    return pc.coalesce(*(batch.column(survey) for survey in SPECTRUM_SURVEYS))


def blocks(batch: pa.RecordBatch | pa.Table) -> np.ndarray:
    spectra = spectrum_cells(batch)
    has_spectrum = pc.is_valid(spectra).to_numpy(zero_copy_only=False)
    has_hsc = pc.is_valid(batch.column("hsc")).to_numpy(zero_copy_only=False)
    has_redshift = pc.is_valid(batch.column(REDSHIFT)).to_numpy(zero_copy_only=False)
    bounds = layout(has_spectrum, has_hsc, has_redshift)
    first = bounds[:-1]
    table_value_first = table_value_starts(first, has_spectrum)
    built = np.empty((bounds[-1], DIM), dtype=np.float32)
    built[positions(first, N_IMAGE_TOKENS)] = image_tokens(batch.column(ANCHOR))
    if has_spectrum.any():
        built[positions(first[has_spectrum] + N_IMAGE_TOKENS, N_SPECTRUM_TOKENS)] = (
            spectral(spectra.drop_null())
        )
    built[positions(table_value_first, N_LS_TABLE_VALUES)] = rows(
        batch.column(ANCHOR), N_IMAGE_TOKENS, None
    )
    if has_hsc.any():
        built[
            positions(
                table_value_first[has_hsc] + N_LS_TABLE_VALUES, N_HSC_TABLE_VALUES
            )
        ] = rows(batch.column("hsc"), N_IMAGE_TOKENS, None)
    if has_redshift.any():
        built[bounds[1:][has_redshift] - 1] = rows(batch.column(REDSHIFT), 0, None)
    return built


@cache
def index() -> faiss.Index:
    loaded = faiss.read_index(str(artifact("search_index")), faiss.IO_FLAG_MMAP)
    loaded.make_direct_map()
    loaded.parallel_mode = 1
    faiss.downcast_InvertedLists(loaded.invlists).prefetch_nthread = 0
    return loaded


def generate_index() -> None:
    dataset = source("encoded")
    columns = [ANCHOR, "hsc", *SPECTRUM_SURVEYS, REDSHIFT]
    galaxies = dataset.count_rows()
    sample = np.random.default_rng(SEED).choice(
        galaxies, min(TRAIN_GALAXIES, galaxies), replace=False
    )
    training = blocks(dataset.take(sample, columns=columns))
    nlist = min(
        NLIST, len(training) // faiss.ClusteringParameters().min_points_per_centroid
    )
    built = faiss.index_factory(DIM, f"IVF{nlist},SQfp16", faiss.METRIC_INNER_PRODUCT)
    built.train(training)
    del training

    for batch in dataset.to_batches(columns=columns, batch_size=BATCH):
        built.add(blocks(batch))

    faiss.write_index(built, str(artifact("search_index")))
