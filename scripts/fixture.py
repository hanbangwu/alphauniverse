import argparse
import os
from itertools import batched
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.config import (
    ANCHOR,
    DIM,
    GEMMA_DIM,
    GRAM_NAMES,
    IMAGE_VOCABULARY,
    KEPT,
    N_IMAGE_TOKENS,
    N_MORPHOLOGIES,
    N_SPECTRUM_TOKENS,
    PREDICTIONS,
    PROBED,
    PROPERTIES,
    REDSHIFT,
    REDSHIFT_VOCABULARY,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_RANK,
    STORE_COLUMNS,
    TABLE_VALUE_SURVEYS,
    TEXT_SURVEYS,
    TOKEN_EMBEDDING_RANK,
    TOP_CODES,
    VOCABULARY,
    artifact,
    build_dir,
    galaxy_count,
    labels,
    points,
    store_schema,
    store_writer,
)
from app.embeddings import generate_token_search
from app.pql import (
    MATRICES,
    SAMPLES,
    SUMS,
    basis,
    observed,
    prediction_basis,
    prediction_batch,
    predictions,
    save_prediction_basis,
    stored,
)
from app.search import source, tokens
from app.text_search import aion_gemma_space

TOKENS: dict[str, int] = {
    ANCHOR: N_IMAGE_TOKENS + 12,
    "hsc": N_IMAGE_TOKENS + 13,
    "desi": 273,
    "sdss": 273,
    REDSHIFT: 1,
}

STRIDE: dict[str, int] = {ANCHOR: 1, "hsc": 2, "desi": 3, "sdss": 4, REDSHIFT: 6}

CLUSTERS = 64
NOISE = 0.05

PREDICTION_BATCH = 5


def covered(survey: str, galaxy: int) -> bool:
    return galaxy % STRIDE[survey] == 0


def _cells(
    rng: np.random.Generator, centres: np.ndarray, galaxies: int
) -> tuple[dict[str, list[np.ndarray | None]], dict[str, list[np.ndarray | None]]]:
    embeddings: dict[str, list[np.ndarray | None]] = {survey: [] for survey in TOKENS}
    token_cells: dict[str, list[np.ndarray | None]] = {survey: [] for survey in TOKENS}

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
    with store_writer(role) as writer:
        writer.write_table(
            pa.table(
                {
                    "galaxy": np.arange(galaxies),
                    **{
                        survey: [
                            None if cell is None else list(cell)
                            for cell in cells[survey]
                        ]
                        for survey in STORE_COLUMNS
                    },
                    **flags,
                },
                schema=store_schema(role),
            )
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _predictions(rng: np.random.Generator, galaxies: int) -> None:
    ones = np.ones((VOCABULARY, 1))
    orthogonal, _ = np.linalg.qr(
        np.hstack((ones, rng.standard_normal((VOCABULARY, SPECTRUM_TOKEN_RANK))))
    )
    save_prediction_basis(
        {
            survey: (np.full(VOCABULARY, 1 / VOCABULARY), orthogonal[:, 1:].T)
            for survey in SPECTRUM_SURVEYS
        }
    )
    records = []
    for galaxy in range(galaxies):
        values = {
            "galaxy": np.asarray([galaxy], dtype=np.int32),
            REDSHIFT: _softmax(3 * rng.standard_normal(REDSHIFT_VOCABULARY)).astype(
                np.float32
            ),
        }
        for survey, table_values in TABLE_VALUE_SURVEYS.items():
            mass = rng.dirichlet(np.full(TOP_CODES + 1, 0.3), size=N_IMAGE_TOKENS)
            codes = (
                rng.random((N_IMAGE_TOKENS, IMAGE_VOCABULARY))
                .argpartition(TOP_CODES, axis=1)[:, :TOP_CODES]
                .astype(np.uint16)
            )
            values[f"{survey}_codes"] = codes[:, :KEPT].reshape(-1)
            values[f"{survey}_query_codes"] = codes[:, KEPT:].reshape(-1)
            probabilities = -np.sort(-mass[:, :TOP_CODES], axis=1)
            values[f"{survey}_probabilities"] = (
                probabilities[:, :KEPT].astype(np.float32).reshape(-1)
            )
            values[f"{survey}_query_probabilities"] = (
                probabilities[:, KEPT:].astype(np.float16).reshape(-1)
            )
            values[f"{survey}_rest"] = (
                probabilities[:, KEPT:].sum(axis=1) + mass[:, -1]
            ).astype(np.float32)
            values[f"{survey}_tails"] = mass[:, -1].astype(np.float32)
            values[f"{survey}_table_values"] = (
                _softmax(3 * rng.standard_normal((len(table_values), VOCABULARY)))
                .astype(np.float32)
                .reshape(-1)
            )
        for survey in SPECTRUM_SURVEYS:
            steps = rng.uniform(1e-6, 3e-6, N_SPECTRUM_TOKENS).astype(np.float32)
            values[f"{survey}_coefficients"] = rng.integers(
                256, size=N_SPECTRUM_TOKENS * SPECTRUM_TOKEN_RANK, dtype=np.uint8
            )
            values[f"{survey}_offsets"] = -128 * steps
            values[f"{survey}_steps"] = steps
        records.append(values)
    with pa.ipc.new_file(artifact("probabilities"), PREDICTIONS) as writer:
        for chunk in batched(records, PREDICTION_BATCH):
            writer.write_batch(prediction_batch(list(chunk)))


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True) @ basis


def _properties(
    rng: np.random.Generator, galaxies: int, provabgs: np.ndarray
) -> pa.Table:
    measured = {f"{label}_measured": provabgs for label in PROBED} | {
        f"{REDSHIFT}_measured": np.array(
            [covered(REDSHIFT, galaxy) for galaxy in range(galaxies)]
        )
    }
    return pa.table(
        {"galaxy": np.arange(galaxies, dtype=np.int32)}
        | {
            name: pa.array(
                rng.random(galaxies, np.float32),
                mask=~measured.get(name, np.ones(galaxies, dtype=bool)),
            )
            for name in PROPERTIES.names[1:]
        },
        schema=PROPERTIES,
    )


def _token_embeddings(rng: np.random.Generator, galaxies: int) -> None:
    for name in GRAM_NAMES:
        np.savez(
            artifact(f"token_embedding_{name}"),
            queries=rng.standard_normal((galaxies, TOKEN_EMBEDDING_RANK), np.float32),
            galleries=rng.standard_normal((galaxies, TOKEN_EMBEDDING_RANK), np.float32),
            offsets=rng.standard_normal(galaxies, np.float32),
            self_scores=rng.standard_normal(galaxies, np.float32),
            singular_values=np.sort(rng.random(galaxies))[::-1],
        )


def forget() -> None:
    for cached in (
        galaxy_count,
        labels,
        source,
        tokens,
        aion_gemma_space,
        predictions,
        prediction_basis,
        basis,
        observed,
        stored,
    ):
        cached.cache_clear()
    MATRICES.clear()
    SAMPLES.clear()
    SUMS.clear()


def build(galaxies: int, seed: int = 0) -> Path:
    target = build_dir()
    target.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((CLUSTERS, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

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

    held = {
        modality: rows[[covered(survey, galaxy) for galaxy in rows]]
        for modality, survey in TEXT_SURVEYS.items()
    }
    galaxy = np.concatenate(list(held.values())).astype(np.int32)
    vectors = rng.standard_normal((len(galaxy), GEMMA_DIM)).astype(np.float32)
    np.savez(
        artifact("aion_gemma_space"),
        vectors=vectors / np.linalg.norm(vectors, axis=1, keepdims=True),
        galaxy=galaxy,
        modality=np.repeat(list(held), [len(members) for members in held.values()]),
    )

    _predictions(rng, galaxies)
    pq.write_table(
        _properties(rng, galaxies, flags["provabgs"]), artifact("properties")
    )
    _token_embeddings(rng, galaxies)
    generate_token_search()

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
