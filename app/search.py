"""Patch-level similarity search over the anchor survey's encoded embeddings.

The index holds one vector per anchor image patch, L2-normalised so that inner
product is cosine similarity. Vector ids encode position:

    faiss id = galaxy * N_PATCHES + patch

which holds because `generate_index()` adds every galaxy's `N_PATCHES`
anchor patches in row order, contiguously. Nothing else in the codebase may add
to or reorder the index without breaking that mapping.
"""

from functools import cache
from typing import Annotated

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import normalize

from .config import (
    ANCHOR,
    BATCH,
    DIM,
    MIN_TRAIN_PER_CENTROID,
    N_PATCHES,
    NLIST,
    NPROBE,
    PROBE,
    SEED,
    TRAIN_GALAXIES,
    GalaxyIndex,
    artifact,
)


class Query(BaseModel):
    """A patch-similarity request: some patches of one galaxy, and a result size."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    galaxy: GalaxyIndex
    patches: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_PATCHES)], ...],
        Field(alias="p", min_length=1, max_length=N_PATCHES),
    ]
    matches: Annotated[int, Field(ge=1, le=128)] = 32


def search(
    query: Query, *, index: faiss.Index
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank galaxies against the mean of the query patches.

    Returns `(galaxies, scores, maps)`: the galaxy ids, each one's best patch
    score, and the full per-patch score map. The query galaxy is always row 0;
    the rest are sorted by descending score.

    Candidates come from one ANN search for the `PROBE` nearest patches, which
    caps the result at however many distinct galaxies those patches belong to —
    so fewer than `query.matches` rows can come back. Scores are then computed
    exactly against every patch of every candidate, so the ordering within the
    candidate set is exact and only the candidate set itself is approximate.
    """
    direction = normalize(
        index.reconstruct_batch(
            query.galaxy * N_PATCHES + np.asarray(query.patches)
        ).mean(axis=0, keepdims=True)
    )

    ids = index.search(direction, PROBE)[1][0]
    found, first = np.unique(ids[ids >= 0] // N_PATCHES, return_index=True)
    ranked = found[np.argsort(first)]
    order = np.concatenate(
        ([query.galaxy], ranked[ranked != query.galaxy][: query.matches])
    ).astype(np.int32)

    maps = (
        index.reconstruct_batch(
            (order[:, None] * N_PATCHES + np.arange(N_PATCHES)).reshape(-1)
        )
        @ direction.T
    ).reshape(len(order), N_PATCHES)
    scores = maps.max(axis=1)
    by_score = np.concatenate(([0], 1 + np.argsort(-scores[1:], kind="stable")))
    return order[by_score], scores[by_score], maps[by_score]


@cache
def source(role: str) -> ds.Dataset:
    """The parquet dataset for `role`, opened once per process."""
    return ds.dataset(artifact(role), format="parquet")


def patches(cells: pa.Array) -> np.ndarray:
    """The image patches of each cell as normalised float32 rows.

    Each cell holds a galaxy's image patches followed by its scalar tokens; the
    slice to `N_PATCHES` keeps only the image patches.
    """
    return normalize(
        np.asarray(
            pc.list_slice(cells, 0, N_PATCHES).flatten().flatten(), dtype=np.float32
        ).reshape(-1, DIM),
        copy=False,
    )


@cache
def index() -> faiss.Index:
    """The search index, read into memory once per process.

    `make_direct_map` is what lets `search()` reconstruct vectors by id.
    """
    loaded = faiss.read_index(str(artifact("encoded_index")))
    loaded.make_direct_map()
    loaded.nprobe = NPROBE
    return loaded


def generate_index(nlist: int | None = None) -> None:
    """Build the search index from `encoded.parquet` and write it to disk.

    Trains on a sample of galaxies, then adds every galaxy's anchor patches in
    row order.

    `nlist` defaults to `NLIST`, capped so training never falls below faiss's
    `MIN_TRAIN_PER_CENTROID` vectors per centroid. At production scale the cap
    is inactive; it is what keeps a small fixture dataset buildable. Pass a value
    to choose the geometry directly.
    """
    dataset = source("encoded")
    galaxies = dataset.count_rows()
    sample = np.random.default_rng(SEED).choice(
        galaxies, min(TRAIN_GALAXIES, galaxies), replace=False
    )
    training = patches(
        dataset.take(sample, columns=[ANCHOR]).column(ANCHOR).combine_chunks()
    )
    if nlist is None:
        nlist = max(1, min(NLIST, len(training) // MIN_TRAIN_PER_CENTROID))

    built = faiss.index_factory(DIM, f"IVF{nlist},SQfp16", faiss.METRIC_INNER_PRODUCT)
    built.train(training)
    for batch in dataset.to_batches(columns=[ANCHOR], batch_size=BATCH):
        built.add(patches(batch.column(ANCHOR)))

    faiss.write_index(built, str(artifact("encoded_index")))
