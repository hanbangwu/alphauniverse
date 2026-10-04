import argparse
import os
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.config import (
    ANCHOR,
    DIM,
    N_MORPHOLOGIES,
    N_PATCHES,
    TOKEN_SURVEYS,
    artifact,
    build_dir,
    galaxy_count,
    labels,
    points,
    store_schema,
)
from app.search import generate_index, index, source, starts, tokens, with_spectrum

TOKENS: dict[str, int] = {
    ANCHOR: N_PATCHES + 12,
    "hsc": N_PATCHES + 13,
    "desi": 273,
    "sdss": 273,
}

STRIDE: dict[str, int] = {ANCHOR: 1, "hsc": 2, "desi": 3, "sdss": 4}

CLUSTERS = 64
NOISE = 0.05


def covered(survey: str, galaxy: int) -> bool:
    return galaxy % STRIDE[survey] == 0


def _cells(
    rng: np.random.Generator, centres: np.ndarray, galaxies: int
) -> tuple[dict[str, list[np.ndarray | None]], dict[str, list[np.ndarray | None]]]:
    embeddings: dict[str, list[np.ndarray | None]] = {
        survey: [] for survey in TOKEN_SURVEYS
    }
    token_cells: dict[str, list[np.ndarray | None]] = {
        survey: [] for survey in TOKEN_SURVEYS
    }

    for galaxy in range(galaxies):
        for survey, count in TOKENS.items():
            if not covered(survey, galaxy):
                embeddings[survey].append(None)
                token_cells[survey].append(None)
                continue
            assigned = rng.integers(len(centres), size=count)
            rows = centres[assigned] + NOISE * rng.standard_normal((count, DIM))
            embeddings[survey].append(rows.astype(np.float16))
            token_cells[survey].append(assigned.astype(np.uint32))

    return embeddings, token_cells


def _store(
    role: str,
    cells: dict[str, list[np.ndarray | None]],
    galaxies: int,
    flags: dict[str, np.ndarray],
) -> None:
    pq.write_table(
        pa.table(
            {
                "galaxy": np.arange(galaxies),
                **{
                    survey: [
                        None if cell is None else list(cell) for cell in cells[survey]
                    ]
                    for survey in TOKEN_SURVEYS
                },
                **flags,
            },
            schema=store_schema(role),
        ),
        artifact(role),
        compression="zstd",
    )


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    unit = rows.copy()
    faiss.normalize_L2(unit)
    return unit @ basis


def forget() -> None:
    for cached in (galaxy_count, labels, source, index, tokens, with_spectrum, starts):
        cached.cache_clear()


def build(galaxies: int, seed: int = 0) -> Path:
    target = build_dir()
    target.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((CLUSTERS, DIM)).astype(np.float32)
    faiss.normalize_L2(centres)

    embeddings, token_cells = _cells(rng, centres, galaxies)

    rows = np.arange(galaxies)
    labelled = rows % 10 != 0
    flags = {"gz10": labelled, "provabgs": rows % 3 != 0}

    _store("encoded", embeddings, galaxies, flags)
    _store("tokens", token_cells, galaxies, flags)

    category = pa.array(rng.integers(N_MORPHOLOGIES, size=galaxies), mask=~labelled)

    anchors = np.stack(
        [np.asarray(cell, dtype=np.float32) for cell in embeddings[ANCHOR]]
    )
    basis = rng.standard_normal((DIM, 2)).astype(np.float32)

    pq.write_table(
        points(rows, _project(basis, anchors.mean(axis=1)), category),
        artifact("mean_points"),
        compression="zstd",
    )

    owner = np.repeat(rows, anchors.shape[1])
    pq.write_table(
        points(owner, _project(basis, anchors.reshape(-1, DIM)), category.take(owner)),
        artifact("full_points"),
        compression="zstd",
    )

    forget()
    generate_index()
    forget()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builds a small artifact tree in the schemas that app/ serves."
    )
    parser.add_argument("--galaxies", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(".cache") / "fixture")
    arguments = parser.parse_args()

    os.environ["ALPHAUNIVERSE_CACHE"] = str(arguments.out.resolve())
    target = build(arguments.galaxies, arguments.seed)

    total = sum(path.stat().st_size for path in target.iterdir())
    print(f"{arguments.galaxies} galaxies -> {target} ({total / 1e6:.1f} MB)")
    for path in sorted(target.iterdir()):
        print(f"  {path.name:24} {path.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
