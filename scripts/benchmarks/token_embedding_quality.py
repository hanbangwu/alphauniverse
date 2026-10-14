import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import modal
import numpy as np

from app.config import DATASET_REVISION, GRAM_NAMES, TOKEN_GRAMS
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-token-embedding-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/token_embedding_quality.json")
RANKS = (8, 16, 32, 64, 128, 256, 512, 1024)
TOP = 100
SHORT = 10
BATCH = 1024
COMBINED = {
    "same_place": tuple(TOKEN_GRAMS),
    "any_position": tuple(names[-1] for names in TOKEN_GRAMS.values()),
}


def ranked(scores, first: int):
    import torch

    scores = scores.clone()
    rows = torch.arange(len(scores), device=scores.device)
    scores[rows, first + rows] = -torch.inf
    return scores.topk(TOP, dim=1).indices


def exact_top(rows: Callable, count: int):
    import torch

    return torch.cat(
        [
            ranked(rows(slice(first, first + BATCH)), first)
            for first in range(0, count, BATCH)
        ]
    )


def agreement(rows: Callable, exact) -> dict[str, float]:
    found = exact_top(rows, len(exact))
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
        for count in (SHORT, TOP)
    }


def matrix_rows(rows: slice, matrix):
    return matrix[rows].float()


def truncated(vectors: dict, rank: int) -> dict:
    return {
        "queries": vectors["queries"][:, :rank],
        "galleries": vectors["galleries"][:, :rank],
        "offsets": vectors["offsets"],
    }


def stored_rows(rows: slice, queries, galleries, offsets):
    from app.embeddings import scores

    return scores(queries[rows], galleries, offsets)


def positive_rows(rows: slice, positions, diagonal):
    return positions[rows] @ positions.T - ((positions**2).sum(dim=1) - diagonal) / 2


def standardised(rows):
    return (rows - rows.mean(dim=1, keepdim=True)) / rows.std(
        dim=1, correction=0, keepdim=True
    )


def combined_rows(rows: slice, parts: list[Callable]):
    return sum(standardised(part(rows)) for part in parts)


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=(32 * 1024, 64 * 1024),
    timeout=15 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    from app import embeddings
    from app.config import artifact

    grams, vectors, exact, matrices = {}, {}, {}, {}
    for name in tqdm(GRAM_NAMES, desc="matrices"):
        gram = embeddings.loaded(name)
        symmetric = (gram + gram.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(embeddings.centred(symmetric)[0])
        eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
        stored = np.load(artifact(f"token_embedding_{name}"))
        vectors[name] = {
            part: torch.from_numpy(stored[part]).to(gram.device)
            for part in ("queries", "galleries", "offsets")
        }
        grams[name] = gram.float()
        exact[name] = exact_top(partial(matrix_rows, matrix=grams[name]), len(gram))
        energy = stored["singular_values"] ** 2
        energy = energy.cumsum() / energy.sum()
        negative = eigenvalues[eigenvalues < 0]
        matrix = {
            "negative_eigenvalues": len(negative),
            "negative_share": round(
                float(negative.abs().sum() / eigenvalues.abs().sum()), 4
            ),
            "energy": {rank: round(float(energy[rank - 1]), 4) for rank in RANKS},
            "stored": {},
            "positive": {},
        }
        for rank in RANKS:
            matrix["stored"][rank] = agreement(
                partial(stored_rows, **truncated(vectors[name], rank)), exact[name]
            )
            positions = eigenvectors[:, :rank] * eigenvalues[:rank].clamp_min(0).sqrt()
            matrix["positive"][rank] = agreement(
                partial(
                    positive_rows,
                    positions=positions.float(),
                    diagonal=grams[name].diagonal(),
                ),
                exact[name],
            )
        if name not in TOKEN_GRAMS:
            matrix["antisymmetric_share"] = round(
                float(
                    ((gram - gram.T) / 2).square().sum()
                    / (gram - gram.mean()).square().sum()
                ),
                4,
            )
            matrix["symmetric"] = agreement(
                partial(matrix_rows, matrix=symmetric), exact[name]
            )
        matrices[name] = matrix
        del gram, symmetric, eigenvalues, eigenvectors
        torch.cuda.empty_cache()

    combined = {}
    for combination, names in tqdm(COMBINED.items(), desc="combined"):
        truth = exact_top(
            partial(
                combined_rows,
                parts=[partial(matrix_rows, matrix=grams[name]) for name in names],
            ),
            len(exact[names[0]]),
        )
        combined[combination] = {
            rank: agreement(
                partial(
                    combined_rows,
                    parts=[
                        partial(stored_rows, **truncated(vectors[name], rank))
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
        "galaxies": len(exact[GRAM_NAMES[0]]),
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
