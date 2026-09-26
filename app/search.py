from functools import cache
from typing import Annotated, Self

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.preprocessing import normalize

from .config import (
    ANCHOR,
    BATCH,
    DIM,
    MIN_TRAIN_PER_CENTROID,
    N_PATCHES,
    N_SPANS,
    NLIST,
    NPROBE,
    PROBE,
    SEED,
    SPECTRUM_SURVEYS,
    TRAIN_GALAXIES,
    GalaxyIndex,
    artifact,
)


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


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(artifact(role), format="parquet")


@cache
def with_spectrum() -> np.ndarray:
    table = source("tokens").to_table(columns=list(SPECTRUM_SURVEYS))
    return np.logical_or.reduce(
        [
            pc.is_valid(table.column(survey)).to_numpy(zero_copy_only=False)
            for survey in SPECTRUM_SURVEYS
        ]
    )


@cache
def starts() -> np.ndarray:
    has = with_spectrum()
    before = np.concatenate(([0], np.cumsum(has)[:-1]))
    return np.arange(len(has)) * N_PATCHES + before * N_SPANS


def owners(ids: np.ndarray) -> np.ndarray:
    return np.searchsorted(starts(), ids, side="right") - 1


def centroid(query: Query, *, index: faiss.Index) -> np.ndarray:
    start = starts()[query.galaxy]
    ids = np.concatenate(
        (
            start + np.asarray(query.patches, dtype=np.int64),
            start + N_PATCHES + np.asarray(query.spans, dtype=np.int64),
        )
    )
    return normalize(index.reconstruct_batch(ids).mean(axis=0, keepdims=True))


def candidates(
    query: Query, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    ids = index.search(direction, PROBE)[1][0]
    found, first = np.unique(owners(ids[ids >= 0]), return_index=True)
    ranked = found[np.argsort(first)]
    return np.concatenate(
        ([query.galaxy], ranked[ranked != query.galaxy][: query.matches])
    ).astype(np.int32)


def vectors(order: np.ndarray, *, index: faiss.Index) -> np.ndarray:
    return index.reconstruct_batch(
        (starts()[order][:, None] + np.arange(N_PATCHES)).reshape(-1)
    )


def score_maps(rows: np.ndarray, direction: np.ndarray, *, width: int) -> np.ndarray:
    return (rows @ direction.T).reshape(-1, width)


def span_maps(
    order: np.ndarray, direction: np.ndarray, *, index: faiss.Index
) -> np.ndarray:
    maps = np.full((len(order), N_SPANS), np.nan, dtype=np.float32)
    has = with_spectrum()[order]
    if has.any():
        rows = index.reconstruct_batch(
            (starts()[order[has]][:, None] + N_PATCHES + np.arange(N_SPANS)).reshape(-1)
        )
        maps[has] = score_maps(rows, direction, width=N_SPANS)
    return maps


def rank(
    order: np.ndarray, scored: np.ndarray, spectral: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.maximum(
        scored.max(axis=1), np.nan_to_num(spectral, nan=-np.inf).max(axis=1)
    )
    by_score = np.concatenate(([0], 1 + np.argsort(-scores[1:], kind="stable")))
    return order[by_score], scores[by_score], scored[by_score], spectral[by_score]


def search(
    query: Query, *, index: faiss.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = centroid(query, index=index)
    order = candidates(query, direction, index=index)
    scored = score_maps(vectors(order, index=index), direction, width=N_PATCHES)
    return rank(order, scored, span_maps(order, direction, index=index))


def rows(cells: pa.Array, start: int, stop: int | None) -> np.ndarray:
    flat = pc.list_flatten(pc.list_flatten(pc.list_slice(cells, start, stop)))
    return normalize(np.asarray(flat, dtype=np.float32).reshape(-1, DIM), copy=False)


def patches(cells: pa.Array) -> np.ndarray:
    return rows(cells, 0, N_PATCHES)


def spectral(cells: pa.Array) -> np.ndarray:
    return rows(cells, 1, None)


def spectrum_cells(batch: pa.RecordBatch | pa.Table) -> pa.Array:
    return pc.coalesce(*(batch.column(survey) for survey in SPECTRUM_SURVEYS))


def blocks(batch: pa.RecordBatch | pa.Table) -> np.ndarray:
    spectra = spectrum_cells(batch)
    has = pc.is_valid(spectra).to_numpy(zero_copy_only=False)
    patch_blocks = patches(batch.column(ANCHOR)).reshape(-1, N_PATCHES, DIM)
    span_blocks = iter(
        spectral(spectra.drop_null()).reshape(-1, N_SPANS, DIM) if has.any() else ()
    )
    return np.concatenate(
        [
            block
            for galaxy, patch_block in enumerate(patch_blocks)
            for block in (
                (patch_block, next(span_blocks)) if has[galaxy] else (patch_block,)
            )
        ]
    )


@cache
def index() -> faiss.Index:
    loaded = faiss.read_index(str(artifact("encoded_index")), faiss.IO_FLAG_MMAP)
    loaded.make_direct_map()
    loaded.nprobe = NPROBE
    return loaded


def generate_index() -> None:
    dataset = source("encoded")
    columns = [ANCHOR, *SPECTRUM_SURVEYS]
    galaxies = dataset.count_rows()
    sample = np.random.default_rng(SEED).choice(
        galaxies, min(TRAIN_GALAXIES, galaxies), replace=False
    )
    training = blocks(dataset.take(sample, columns=columns))
    nlist = max(1, min(NLIST, len(training) // MIN_TRAIN_PER_CENTROID))
    built = faiss.index_factory(DIM, f"IVF{nlist},SQfp16", faiss.METRIC_INNER_PRODUCT)
    built.train(training)
    del training

    for batch in dataset.to_batches(columns=columns, batch_size=BATCH):
        built.add(blocks(batch))

    faiss.write_index(built, str(artifact("encoded_index")))
