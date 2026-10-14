import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import modal
import numpy as np

from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    TOKEN_EMBEDDING_RANK,
    TOKEN_GRAMS,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-token-embedding-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/token_embedding_quality.json")
RANKS = tuple(TOKEN_EMBEDDING_RANK // 2**halvings for halvings in range(7, -1, -1))
TOP = (10, 100)
BATCH = 1024
COMBINED = {
    "same_place": tuple(TOKEN_GRAMS),
    "any_position": tuple(names[-1] for names in TOKEN_GRAMS.values()),
}


def top(score: Callable, count: int):
    import torch

    found = []
    for first in range(0, count, BATCH):
        scores = score(slice(first, first + BATCH)).clone()
        scores.diagonal(first).fill_(-torch.inf)
        found.append(scores.topk(max(TOP), dim=1).indices)
    return torch.cat(found)


def agreement(score: Callable, exact) -> dict[str, float]:
    found = top(score, len(exact))
    return {
        f"top{count}": round(
            float(
                (found[:, :count, None] == exact[:, None, :count])
                .any(-1)
                .float()
                .mean()
            ),
            4,
        )
        for count in TOP
    }


def matrix_rows(rows: slice, matrix):
    return matrix[rows].float()


def stored_rows(rows: slice, vectors: dict, rank: int):
    return (
        vectors["queries"][rows, :rank] @ vectors["galleries"][:, :rank].T
        + vectors["offsets"]
    )


def positive_rows(rows: slice, positions, diagonal):
    return positions[rows] @ positions.T - ((positions**2).sum(dim=1) - diagonal) / 2


def standardised(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, correction=0, keepdim=True
    )


def combined_rows(rows: slice, parts: list[Callable]):
    return sum(standardised(part(rows)) for part in parts)


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=8 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    from app import embeddings
    from app.config import artifact

    grams, vectors, matrices = {}, {}, {}
    for name in tqdm(GRAM_NAMES, desc="matrices"):
        gram = embeddings.loaded(name)
        symmetric = (gram + gram.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(embeddings.centred(symmetric))
        positions = (
            eigenvectors[:, -TOKEN_EMBEDDING_RANK:]
            * eigenvalues[-TOKEN_EMBEDDING_RANK:].clamp_min(0).sqrt()
        )
        positions = positions.flip(1).float()
        stored = np.load(artifact(f"token_embedding_{name}"))
        vectors[name] = {
            part: torch.from_numpy(stored[part]).to(gram.device)
            for part in ("queries", "galleries", "offsets")
        }
        grams[name] = gram.float()
        exact = top(partial(matrix_rows, matrix=grams[name]), len(gram))
        energy = stored["singular_values"] ** 2
        energy = energy.cumsum() / energy.sum()
        negative = eigenvalues[eigenvalues < 0]
        matrices[name] = {
            "negative_eigenvalues": len(negative),
            "negative_share": round(
                float(negative.abs().sum() / eigenvalues.abs().sum()), 4
            ),
            "energy": {rank: round(float(energy[rank - 1]), 4) for rank in RANKS},
            "stored": {
                rank: agreement(
                    partial(stored_rows, vectors=vectors[name], rank=rank), exact
                )
                for rank in RANKS
            },
            "positive": {
                rank: agreement(
                    partial(
                        positive_rows,
                        positions=positions[:, :rank],
                        diagonal=grams[name].diagonal(),
                    ),
                    exact,
                )
                for rank in RANKS
            },
        }
        if name not in TOKEN_GRAMS:
            matrices[name]["antisymmetric_share"] = round(
                float(
                    ((gram - gram.T) / 2).square().sum()
                    / (gram - gram.mean()).square().sum()
                ),
                4,
            )
            matrices[name]["symmetric"] = agreement(
                partial(matrix_rows, matrix=symmetric), exact
            )
        del gram, symmetric, eigenvalues, eigenvectors

    galaxies = len(grams[GRAM_NAMES[0]])
    combined = {}
    for combination, names in tqdm(COMBINED.items(), desc="combined"):
        truth = top(
            partial(
                combined_rows,
                parts=[partial(matrix_rows, matrix=grams[name]) for name in names],
            ),
            galaxies,
        )
        combined[combination] = {
            rank: agreement(
                partial(
                    combined_rows,
                    parts=[
                        partial(stored_rows, vectors=vectors[name], rank=rank)
                        for name in names
                    ],
                ),
                truth,
            )
            for rank in RANKS
        }

    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": galaxies,
        "ranks": RANKS,
        "matrices": matrices,
        "combined": combined,
        "memory": memory(),
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **measure.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
