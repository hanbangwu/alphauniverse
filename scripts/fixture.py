"""Builds a small artifact tree carrying the shapes that app/ serves.

Real artifacts come from the Modal build jobs: a GPU, the AION encoder and tens
of gigabytes of embeddings. Neither the test suite nor CI can do that, so this
writes artifacts with the production schemas at a size that fits in a runner.
Everything downstream of the artifacts — the API, the search path, the
benchmarks — then exercises real code against real files.

Embeddings are drawn from a fixed set of random cluster centres rather than
uniform noise. In 768 dimensions uniform random vectors are all near-orthogonal,
which would make every ranking arbitrary and every recall number meaningless.

The tree deliberately omits the `codebook` and `parametric_umap` artifacts:
nothing served depends on them, and their absence exercises the 404 path.

    uv run python -m scripts.fixture --galaxies 12 --out .cache/fixture
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize

from app.config import (
    ANCHOR,
    DIM,
    N_MORPHOLOGIES,
    N_PATCHES,
    POINTS,
    TOKEN_SURVEYS,
    artifact,
    build_dir,
    store_schema,
)
from app.search import generate_index, source

# Token counts per survey, matching app/encode.py: image or spectrum tokens
# first, then that survey's scalars (12 for the anchor, 13 for HSC).
TOKENS: dict[str, int] = {
    ANCHOR: N_PATCHES + 12,
    "hsc": N_PATCHES + 13,
    "desi": 273,
    "sdss": 273,
}

# Which galaxies a survey covers, by row index. The anchor covers everything;
# the rest are strided so coverage is predictable in tests.
STRIDE: dict[str, int] = {ANCHOR: 1, "hsc": 2, "desi": 3, "sdss": 4}

CLUSTERS = 64
NOISE = 0.35


def covered(survey: str, galaxy: int) -> bool:
    """Whether `survey` has a crossmatch for this galaxy."""
    return galaxy % STRIDE[survey] == 0


def _cells(
    rng: np.random.Generator, centres: np.ndarray, galaxies: int
) -> tuple[dict[str, list[np.ndarray | None]], dict[str, list[np.ndarray | None]]]:
    """Per-survey embedding and token-id cells, with `None` where uncovered."""
    embeddings: dict[str, list[np.ndarray | None]] = {s: [] for s in TOKEN_SURVEYS}
    tokens: dict[str, list[np.ndarray | None]] = {s: [] for s in TOKEN_SURVEYS}

    for galaxy in range(galaxies):
        for survey, count in TOKENS.items():
            if not covered(survey, galaxy):
                embeddings[survey].append(None)
                tokens[survey].append(None)
                continue
            assigned = rng.integers(len(centres), size=count)
            rows = centres[assigned] + NOISE * rng.standard_normal((count, DIM))
            embeddings[survey].append(rows.astype(np.float16))
            # Token ids track the cluster, mirroring the real store where a
            # token id and its codebook vector are two views of one thing.
            tokens[survey].append(assigned.astype(np.uint32))

    return embeddings, tokens


def _column(entries: list[np.ndarray | None], dim: int | None) -> pa.Array:
    """A nullable list column over per-galaxy cells, as app/encode.py writes it."""
    present = [entry for entry in entries if entry is not None]
    values = pa.array(
        np.concatenate([entry.reshape(-1) for entry in present])
        if present
        else np.empty(0, np.uint32 if dim is None else np.float16)
    )
    offsets = pa.array(
        np.cumsum([0, *(0 if entry is None else entry.size // (dim or 1) for entry in entries)]),
        type=pa.int32(),
        mask=np.array([*(entry is None for entry in entries), False]),
    )
    return pa.ListArray.from_arrays(
        offsets,
        values if dim is None else pa.FixedSizeListArray.from_arrays(values, dim),
    )


def _store(
    role: str,
    cells: dict[str, list[np.ndarray | None]],
    galaxies: int,
    dim: int | None,
    flags: dict[str, np.ndarray],
) -> None:
    """Write one parquet store, in the schema app/encode.py writes."""
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array(np.arange(galaxies), type=pa.int32()),
                **{s: _column(cells[s], dim) for s in TOKEN_SURVEYS},
                **{s: pa.array(values, type=pa.bool_()) for s, values in flags.items()},
            },
            schema=store_schema(role),
        ),
        artifact(role),
        compression="zstd",
    )


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Project normalised rows onto `basis`.

    Stands in for the trained parametric UMAP, which would pull torch and a
    training run into the test path. Coordinates are structured but arbitrary.

    One basis is drawn per build and reused for both point sets, because the
    real projector is a single function applied to everything: `mean_points` and
    `full_points` have to be coordinates in one shared space.
    """
    return (normalize(rows) @ basis).astype(np.float32)


def build(galaxies: int = 12, seed: int = 0, nlist: int | None = None) -> Path:
    """Write a complete fixture tree under `build_dir()`.

    Returns the directory written. Requires `ALPHAUNIVERSE_CACHE` to already
    point where the tree should go. `nlist` overrides the index geometry that
    `generate_index` would otherwise choose for this size.
    """
    target = build_dir()
    target.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((CLUSTERS, DIM))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    embeddings, tokens = _cells(rng, centres, galaxies)

    # A morphology label and a GZ10 crossmatch are one fact in production: the
    # category is null exactly where the galaxy has no gz10 match. A tenth are
    # unlabelled, so the null path is covered. PROVABGS is an unrelated
    # catalogue and varies independently.
    rows = np.arange(galaxies)
    labelled = rows % 10 != 0
    flags = {"gz10": labelled, "provabgs": rows % 3 != 0}

    _store("encoded", embeddings, galaxies, DIM, flags)
    _store("tokens", tokens, galaxies, None, flags)

    raw = rng.integers(0, N_MORPHOLOGIES, size=galaxies).astype(np.uint8)
    category = pa.array(raw, mask=~labelled, type=pa.uint8())

    anchors = np.stack(
        [np.asarray(cell, dtype=np.float32) for cell in embeddings[ANCHOR]]
    )
    basis = rng.standard_normal((DIM, 2)).astype(np.float32)

    means = _project(basis, anchors.mean(axis=1))
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array(np.arange(galaxies), type=pa.int32()),
                "x": pa.array(means[:, 0]),
                "y": pa.array(means[:, 1]),
                "category": category,
            },
            schema=POINTS,
        ),
        artifact("mean_points"),
        compression="zstd",
    )

    owner = np.repeat(np.arange(galaxies, dtype=np.int32), anchors.shape[1])
    coords = _project(basis, anchors.reshape(-1, DIM))
    pq.write_table(
        pa.table(
            {
                "galaxy": pa.array(owner),
                "x": pa.array(coords[:, 0]),
                "y": pa.array(coords[:, 1]),
                "category": category.take(pa.array(owner)),
            },
            schema=POINTS,
        ),
        artifact("full_points"),
        compression="zstd",
    )

    source.cache_clear()
    # No nlist by default: generate_index's own cap is what makes a small tree
    # buildable, and letting it apply is what keeps that path exercised.
    generate_index(nlist)
    source.cache_clear()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--galaxies", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(".cache") / "fixture")
    arguments = parser.parse_args()

    os.environ["ALPHAUNIVERSE_CACHE"] = str(arguments.out.resolve())
    target = build(arguments.galaxies, arguments.seed, arguments.nlist)

    total = sum(path.stat().st_size for path in target.iterdir())
    print(f"{arguments.galaxies} galaxies -> {target} ({total / 1e6:.1f} MB)")
    for path in sorted(target.iterdir()):
        print(f"  {path.name:24} {path.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
