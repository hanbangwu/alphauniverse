import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
from tqdm import tqdm

from core.artifacts import artifact
from core.config import DATASET_REVISION, GRAM_NAMES, TOKEN_GRAMS
from core.device import device
from core.modal_resources import CACHE_PATH, cache_volume
from pipeline import embeddings
from pipeline.config import TOKEN_EMBEDDING_RANK
from pipeline.modal_app import build_image
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-aion_prediction-embedding-quality")
image = build_image.add_local_python_source("app")

REPORT = Path("docs/benchmarks/aion_prediction_embedding_quality.json")
RANKS = tuple(TOKEN_EMBEDDING_RANK // 2**halvings for halvings in range(7, -1, -1))
TOP = (10, 100)
BATCH = 1024


def top(score: Callable, own: torch.Tensor) -> torch.Tensor:
    found = []
    for first in range(0, len(own), BATCH):
        rows = slice(first, first + BATCH)
        scores = score(rows).clone()
        mine = own[rows]
        listed = mine >= 0
        scores[listed, mine[listed]] = -torch.inf
        found.append(scores.topk(max(TOP), dim=1).indices)
    return torch.cat(found)


def agreement(found: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
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


def own_ranks(score: Callable, own: torch.Tensor) -> dict[str, float]:
    ranks = []
    for part in torch.nonzero(own >= 0).squeeze(1).split(BATCH):
        scores = score(part)
        ranks.append((scores > scores.gather(1, own[part, None])).sum(dim=1) + 1)
    ranks = torch.cat(ranks).double()
    return {
        "p50": float(ranks.median()),
        "p95": float(ranks.quantile(0.95)),
        "top10": round(float((ranks <= 10).double().mean()), 4),
    }


def score_rows(rows: slice | torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix[rows]


def share_rows(
    rows: slice | torch.Tensor, matrix: torch.Tensor, saturation: torch.Tensor
) -> torch.Tensor:
    return matrix[rows] / saturation


def fit_rows(
    rows: slice | torch.Tensor, matrix: torch.Tensor, saturation: torch.Tensor
) -> torch.Tensor:
    return matrix[rows] - saturation


def aion_prediction_rows(
    rows: slice | torch.Tensor, matrix: torch.Tensor, galleries: torch.Tensor
) -> torch.Tensor:
    return matrix[rows][:, galleries]


def stored_rows(
    rows: slice | torch.Tensor,
    queries: torch.Tensor,
    galleries: torch.Tensor,
    offsets: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    return queries[rows, :rank] @ galleries[:, :rank].T + offsets


def block(name: str, mode: str, references: dict) -> dict[str, Any]:
    galleries = torch.from_numpy(references[f"{mode}_galleries"]).to(device())
    saturation = torch.from_numpy(references[f"{mode}_saturation"]).float().to(device())
    measured = embeddings.loaded(f"data_gram_{name}", torch.float32)
    expected = embeddings.loaded(f"aion_prediction_gram_{name}", torch.float32)
    stored = np.load(artifact(f"aion_prediction_embedding_{name}"))
    queries = torch.from_numpy(stored["queries"]).to(device())
    vectors, offsets = (
        torch.from_numpy(stored[part]).to(device())[galleries]
        for part in ("galleries", "offsets")
    )
    own = torch.full((len(measured),), -1, device=device())
    own[galleries] = torch.arange(len(galleries), device=device())
    truths = {"score": partial(score_rows, matrix=measured)}
    if name == mode:
        truths |= {
            "share": partial(share_rows, matrix=measured, saturation=saturation),
            "fit": partial(fit_rows, matrix=measured, saturation=saturation),
        }
    exact = {label: top(score, own) for label, score in truths.items()}
    exact["aion_prediction"] = top(
        partial(aion_prediction_rows, matrix=expected, galleries=galleries), own
    )
    energy = stored["singular_values"] ** 2
    energy = energy.cumsum() / energy.sum()
    agreements = {}
    for rank in RANKS:
        found = top(
            partial(
                stored_rows,
                queries=queries,
                galleries=vectors,
                offsets=offsets,
                rank=rank,
            ),
            own,
        )
        agreements[rank] = {
            label: agreement(found, truth) for label, truth in exact.items()
        }
    return {
        "galleries": len(galleries),
        "own_rank": {label: own_ranks(score, own) for label, score in truths.items()},
        "aion_prediction_gram": {
            label: agreement(exact["aion_prediction"], exact[label]) for label in truths
        },
        "energy": {rank: round(float(energy[rank - 1]), 4) for rank in RANKS},
        "stored": agreements,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=20 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    with np.load(artifact("observed_reference")) as stored:
        references = dict(stored)
    blocks = {
        name: block(name, mode, references)
        for mode, names in tqdm(TOKEN_GRAMS.items(), desc="blocks")
        for name in names
    }
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": len(np.load(artifact(f"data_gram_{GRAM_NAMES[0]}"), mmap_mode="r")),
        "ranks": RANKS,
        "blocks": blocks,
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
