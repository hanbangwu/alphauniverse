import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from umap import UMAP

from app import embeddings
from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    REDSHIFT,
    SEED,
    artifact,
    galaxy_count,
)
from app.parametric_umap import scan
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks import projection_quality
from scripts.benchmarks.common import COMBINED, environment, git, memory, standardised
from scripts.benchmarks.probe_quality import JOINT, RANK, scaled, targets

app = modal.App("alphauniverse-neighbour-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/neighbour_quality.json")
ROLES = ("token", "template")
NEIGHBOURS = 10
QUERIES = 2000
BATCH = 1024
SAME_PLACE = COMBINED["same_place"]
COMBINATIONS = COMBINED | {
    "same_place_without_redshift": tuple(
        name for name in SAME_PLACE if name != REDSHIFT
    )
}


def vectors(role: str) -> dict[str, dict[str, torch.Tensor]]:
    found = {}
    for name in GRAM_NAMES:
        with np.load(artifact(f"{role}_embedding_{name}")) as stored:
            found[name] = {
                part: torch.from_numpy(np.ascontiguousarray(stored[part][:, :RANK]))
                for part in ("queries", "galleries")
            } | {"offsets": torch.from_numpy(stored["offsets"])}
    return found


def joint(role: str) -> torch.Tensor:
    blocks = np.concatenate(
        [scaled(f"{role}_embedding", name) for name in GRAM_NAMES], 1
    )
    return torch.from_numpy(PCA(JOINT, random_state=SEED).fit_transform(blocks))


def searched(
    blocks: dict[str, dict[str, torch.Tensor]],
    names: tuple[str, ...],
    queries: np.ndarray,
) -> torch.Tensor:
    return sum(
        standardised(
            embeddings.scores(
                blocks[name]["queries"][queries], blocks[name], RANK
            ).double()
        )
        for name in names
    )


def top(
    scores: torch.Tensor, sources: np.ndarray, count: int
) -> torch.return_types.topk:
    scores = scores.clone()
    scores[torch.arange(len(sources)), torch.from_numpy(sources)] = -torch.inf
    return scores.topk(count, dim=1)


def overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    shared = (first[:, :, None] == second[:, None, :]).any(dim=2).sum(dim=1)
    return round(float(shared.double().mean() / first.shape[1]), 4)


def physics(
    families: dict[str, dict], joints: dict[str, torch.Tensor], aion: np.ndarray
) -> dict[str, Any]:
    values = targets()
    labelled = np.flatnonzero(np.all([np.isfinite(v) for v in values.values()], 0))
    values = {name: column[labelled] for name, column in values.items()}
    own = np.arange(len(labelled))
    rankings = {}
    for role, blocks in families.items():
        subset = {
            name: {part: block[part][labelled] for part in block}
            for name, block in blocks.items()
        }
        for combination, names in COMBINATIONS.items():
            scores = searched(subset, names, own)
            rankings[f"{role}_{combination}"] = top(scores, own, NEIGHBOURS).indices
        found = joints[role][labelled]
        rankings[f"{role}_joint"] = top(
            -torch.cdist(found, found), own, NEIGHBOURS
        ).indices
    unit = torch.nn.functional.normalize(
        torch.from_numpy(aion[labelled]).double(), dim=1
    )
    rankings["aion"] = top(unit @ unit.T, own, NEIGHBOURS).indices
    rng = np.random.default_rng(SEED)
    rankings["random"] = np.stack(
        [rng.choice(np.delete(own, row), NEIGHBOURS, replace=False) for row in own]
    )
    differences = {
        name: {
            target: np.abs(column[np.asarray(found)] - column[:, None]).mean(axis=1)
            for target, column in values.items()
        }
        for name, found in rankings.items()
    }
    return {
        "galaxies": len(labelled),
        "rankings": {
            name: {
                target: {
                    "median": round(float(np.median(found)), 4),
                    "beats_random": round(
                        float(np.mean(found < differences["random"][target])), 4
                    ),
                }
                for target, found in entry.items()
            }
            for name, entry in differences.items()
        },
    }


def geometry(
    families: dict[str, dict], joints: dict[str, torch.Tensor]
) -> dict[str, Any]:
    count = galaxy_count()
    chosen = np.sort(
        np.random.default_rng(SEED).choice(count, min(QUERIES, count), replace=False)
    )
    found = {}
    for role, blocks in families.items():
        searches, overlaps = {}, {}
        for name in GRAM_NAMES:
            galleries = blocks[name]["galleries"]
            searches[name] = searched(blocks, (name,), chosen)
            overlaps[name] = overlap(
                top(searches[name], chosen, NEIGHBOURS).indices,
                top(
                    -torch.cdist(galleries[chosen], galleries), chosen, NEIGHBOURS
                ).indices,
            )
        nearest = top(
            -torch.cdist(joints[role][chosen], joints[role]), chosen, NEIGHBOURS
        ).indices
        found[role] = {
            "blocks": overlaps,
            "joint": {
                combination: overlap(
                    top(
                        sum(searches[name] for name in names), chosen, NEIGHBOURS
                    ).indices,
                    nearest,
                )
                for combination, names in COMBINED.items()
            },
        }
    return {
        "queries": len(chosen),
        "random": round(NEIGHBOURS / (count - 1), 6),
    } | found


def nearest_by_search(
    blocks: dict[str, dict], count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    everyone = np.arange(galaxy_count())
    parts = [
        top(searched(blocks, SAME_PLACE, rows), rows, count)
        for rows in tqdm(
            np.array_split(everyone, -(-len(everyone) // BATCH)), desc="search"
        )
    ]
    return (
        torch.cat([part.values for part in parts]),
        torch.cat([part.indices for part in parts]),
    )


def search_map(
    found: tuple[torch.Tensor, torch.Tensor], features: np.ndarray
) -> np.ndarray:
    others = UMAP().n_neighbors - 1
    scores, indices = (part[:, :others].numpy() for part in found)
    own = np.arange(len(features))[:, None]
    knn = (
        np.hstack([own, indices]),
        np.hstack([np.zeros(own.shape), 1 + scores[:, :1] - scores]).astype(np.float32),
    )
    return UMAP(random_state=SEED, precomputed_knn=knn).fit_transform(features)


def maps(
    families: dict[str, dict], joints: dict[str, torch.Tensor], aion: np.ndarray
) -> dict[str, Any]:
    largest = max(projection_quality.NEIGHBOURS)
    searches = {
        role: nearest_by_search(blocks, largest) for role, blocks in families.items()
    }
    nearest = {f"{role}_same_place": found[1] for role, found in searches.items()}
    nearest["aion"] = torch.from_numpy(
        NearestNeighbors(n_neighbors=largest, metric="cosine")
        .fit(aion)
        .kneighbors(return_distance=False)
    )
    stored = pq.read_table(artifact("mean_points"), columns=["x", "y"])
    projected = {
        "token_search": search_map(searches["token"], joints["token"].numpy()),
        "token_joint": UMAP(random_state=SEED).fit_transform(joints["token"].numpy()),
        "template_joint": UMAP(random_state=SEED).fit_transform(
            joints["template"].numpy()
        ),
        "site": np.column_stack([stored["x"].to_numpy(), stored["y"].to_numpy()]),
    }
    found = {}
    for name, flat in projected.items():
        near = NearestNeighbors(n_neighbors=largest).fit(flat)
        kept = torch.from_numpy(near.kneighbors(return_distance=False))
        found[name] = {
            search: {
                str(count): overlap(indices[:, :count], kept[:, :count])
                for count in projection_quality.NEIGHBOURS
            }
            for search, indices in nearest.items()
        }
    return found


@app.function(
    image=image,
    cpu=8,
    memory=(16 * 1024, 32 * 1024),
    timeout=30 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    families = {role: vectors(role) for role in ROLES}
    joints = {role: joint(role) for role in ROLES}
    aion = scan(galaxy_count(), np.empty(0, dtype=np.intp))[0]
    return {
        "environment": environment(),
        "rank": RANK,
        "joint": JOINT,
        "neighbours": NEIGHBOURS,
        "preserved": projection_quality.NEIGHBOURS,
        "physics": physics(families, joints, aion),
        "geometry": geometry(families, joints),
        "maps": maps(families, joints, aion),
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
