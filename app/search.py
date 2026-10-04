from functools import cache
from typing import Annotated, Self

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    ANCHOR,
    DIM,
    N_PATCHES,
    N_SPANS,
    SEED,
    SPECTRUM_SURVEYS,
    GalaxyIndex,
    artifact,
)

BATCH = 256
NLIST = 16384
NPROBE = 64
PROBE = 2048
TRAIN_GALAXIES = 2048


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    galaxy: GalaxyIndex
    patches: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_PATCHES)], ...],
        Field(alias="p", max_length=N_PATCHES),
    ] = ()
    spans: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_SPANS)], ...],
        Field(alias="s", max_length=N_SPANS),
    ] = ()
    matches: Annotated[int, Field(ge=1, le=128)] = 32

    @model_validator(mode="after")
    def some_tokens(self) -> Self:
        if not (self.patches or self.spans):
            raise ValueError("select at least one patch or span")
        return self

    @model_validator(mode="after")
    def spans_need_a_spectrum(self) -> Self:
        if self.spans and not with_spectrum()[self.galaxy]:
            raise ValueError(f"galaxy {self.galaxy} has no spectrum")
        return self


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(artifact(role), format="parquet")


@cache
def tokens() -> pa.Table:
    return source("tokens").to_table()


@cache
def with_spectrum() -> np.ndarray:
    return pc.is_valid(spectrum_cells(tokens())).to_numpy()


def layout(has: np.ndarray) -> np.ndarray:
    before = np.concatenate(([0], np.cumsum(has)[:-1]))
    return np.arange(len(has)) * N_PATCHES + before * N_SPANS


@cache
def starts() -> np.ndarray:
    return layout(with_spectrum())


def positions(first: np.ndarray, width: int) -> np.ndarray:
    return (first[:, None] + np.arange(width)).reshape(-1)


def centroid(query: Query, *, index: faiss.Index) -> np.ndarray:
    start = starts()[query.galaxy]
    ids = np.concatenate(
        (
            start + np.asarray(query.patches, dtype=np.int64),
            start + N_PATCHES + np.asarray(query.spans, dtype=np.int64),
        )
    )
    direction = index.reconstruct_batch(ids).mean(axis=0, keepdims=True)
    faiss.normalize_L2(direction)
    return direction


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
    return index.reconstruct_batch(positions(starts()[order], N_PATCHES))


def score_maps(rows: np.ndarray, direction: np.ndarray, *, width: int) -> np.ndarray:
    return (rows @ direction.T).reshape(-1, width)


def span_maps(
    order: np.ndarray, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    maps = np.full((len(order), N_SPANS), np.nan, dtype=np.float32)
    has = with_spectrum()[order]
    rows = index.reconstruct_batch(positions(starts()[order[has]] + N_PATCHES, N_SPANS))
    maps[has] = score_maps(rows, direction, width=N_SPANS)
    return maps


def rank(
    order: np.ndarray, scored: np.ndarray, spectral: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.fmax(scored.max(axis=1), np.fmax.reduce(spectral, axis=1))
    by_score = np.concatenate(([0], 1 + np.argsort(-scores[1:], kind="stable")))
    return order[by_score], scores[by_score], scored[by_score], spectral[by_score]


def search(
    query: Query, *, index: faiss.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = centroid(query, index=index)
    order = candidates(query, direction, index=index)
    scored = score_maps(vectors(order, index=index), direction, width=N_PATCHES)
    return rank(order, scored, span_maps(order, direction, index=index))


def rows(cells: pa.Array | pa.ChunkedArray, start: int, stop: int | None) -> np.ndarray:
    flat = pc.list_flatten(pc.list_flatten(pc.list_slice(cells, start, stop)))
    embeddings = np.asarray(flat, dtype=np.float32).reshape(-1, DIM)
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    faiss.normalize_L2(embeddings)
    return embeddings


def patches(cells: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return rows(cells, 0, N_PATCHES)


def spectral(cells: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return rows(cells, 1, None)


def spectrum_cells(batch: pa.RecordBatch | pa.Table) -> pa.Array | pa.ChunkedArray:
    return pc.coalesce(*(batch.column(survey) for survey in SPECTRUM_SURVEYS))


def blocks(batch: pa.RecordBatch | pa.Table) -> np.ndarray:
    spectra = spectrum_cells(batch)
    has = pc.is_valid(spectra).to_numpy(zero_copy_only=False)
    first = layout(has)
    built = np.empty(
        (len(has) * N_PATCHES + has.sum() * N_SPANS, DIM), dtype=np.float32
    )
    built[positions(first, N_PATCHES)] = patches(batch.column(ANCHOR))
    if has.any():
        built[positions(first[has] + N_PATCHES, N_SPANS)] = spectral(
            spectra.drop_null()
        )
    return built


@cache
def index() -> faiss.Index:
    loaded = faiss.read_index(str(artifact("search_index")), faiss.IO_FLAG_MMAP)
    loaded.make_direct_map()
    return loaded


def generate_index() -> None:
    dataset = source("encoded")
    columns = [ANCHOR, *SPECTRUM_SURVEYS]
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
