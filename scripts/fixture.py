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
    GEMMA_DIM,
    IMAGE_VOCABULARY,
    N_MORPHOLOGIES,
    N_PATCHES,
    N_SPANS,
    PREDICTIONS,
    SCALAR_SURVEYS,
    SPAN_RANK,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
    TOP_CODES,
    VOCABULARY,
    artifact,
    build_dir,
    galaxy_count,
    labels,
    points,
    store_schema,
)
from app.pql import basis, predictions
from app.search import (
    generate_index,
    index,
    source,
    starts,
    tokens,
    with_hsc,
    with_spectrum,
)
from app.text_search import aion_gemma_space

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


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _predictions(rng: np.random.Generator, galaxies: int) -> None:
    ones = np.ones((VOCABULARY, 1))
    orthogonal, _ = np.linalg.qr(
        np.hstack((ones, rng.standard_normal((VOCABULARY, SPAN_RANK))))
    )
    fitted = {
        survey: (np.full(VOCABULARY, 1 / VOCABULARY), orthogonal[:, 1:].T)
        for survey in SPECTRUM_SURVEYS
    }
    np.savez(
        artifact("prediction_basis"),
        **{
            f"{survey}_{name}": value
            for survey, pair in fitted.items()
            for name, value in zip(("mean", "directions"), pair, strict=True)
        },
    )
    columns: dict[str, list[np.ndarray]] = {field.name: [] for field in PREDICTIONS}
    for galaxy in range(galaxies):
        columns["galaxy"].append(np.asarray([galaxy], dtype=np.int32))
        for survey, scalars in SCALAR_SURVEYS.items():
            mass = rng.dirichlet(np.full(TOP_CODES + 1, 0.3), size=N_PATCHES)
            columns[f"{survey}_codes"].append(
                rng.random((N_PATCHES, IMAGE_VOCABULARY))
                .argpartition(TOP_CODES, axis=1)[:, :TOP_CODES]
                .astype(np.uint16)
                .reshape(-1)
            )
            columns[f"{survey}_log_probabilities"].append(
                np.log(-np.sort(-mass[:, :TOP_CODES], axis=1))
                .astype(np.float16)
                .reshape(-1)
            )
            columns[f"{survey}_tails"].append(np.log(mass[:, -1]).astype(np.float32))
            columns[f"{survey}_scalars"].append(
                _log_softmax(3 * rng.standard_normal((len(scalars), VOCABULARY)))
                .astype(np.float16)
                .reshape(-1)
            )
        for survey in SPECTRUM_SURVEYS:
            steps = rng.uniform(1e-6, 3e-6, N_SPANS).astype(np.float32)
            columns[f"{survey}_coefficients"].append(
                rng.integers(256, size=N_SPANS * SPAN_RANK, dtype=np.uint8)
            )
            columns[f"{survey}_offsets"].append(-128 * steps)
            columns[f"{survey}_steps"].append(steps)
    with pa.ipc.new_file(artifact("predictions"), PREDICTIONS) as writer:
        writer.write_batch(
            pa.record_batch(
                [
                    pa.FixedSizeListArray.from_arrays(
                        np.concatenate(columns[field.name]), field.type.list_size
                    )
                    if pa.types.is_fixed_size_list(field.type)
                    else pa.array(np.concatenate(columns[field.name]))
                    for field in PREDICTIONS
                ],
                schema=PREDICTIONS,
            )
        )


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    unit = rows.copy()
    faiss.normalize_L2(unit)
    return unit @ basis


def forget() -> None:
    for cached in (
        galaxy_count,
        labels,
        source,
        index,
        tokens,
        with_spectrum,
        with_hsc,
        starts,
        aion_gemma_space,
        predictions,
        basis,
    ):
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

    vectors = rng.standard_normal((galaxies, GEMMA_DIM)).astype(np.float32)
    np.save(
        artifact("aion_gemma_space"),
        vectors / np.linalg.norm(vectors, axis=1, keepdims=True),
    )

    _predictions(rng, galaxies)

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
