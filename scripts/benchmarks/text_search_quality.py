import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
from scipy.special import logsumexp, softmax
from scipy.stats import bootstrap, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from app.alignment import held_out, pairs
from app.config import (
    DATASET_REVISION,
    REDSHIFT,
    SEED,
    TEXT_SURVEYS,
    VIEW_TYPES,
    galaxy_count,
)
from app.embeddings import loaded
from app.observations import measured_redshifts
from app.properties import provabgs
from app.text_flow import generated_clouds, stored_flow
from app.text_search import (
    SHARE_NAMES,
    aion_gemma_space,
    breadth,
    embed_queries,
    generic_queries,
    indexed,
    likelihoods,
    mode_shares,
    text_index,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-text-search")
image = build_image.add_local_python_source("modal_app")

LABELLED = {
    "a massive galaxy": ("mass", 11, np.inf),
    "a dwarf galaxy": ("mass", -np.inf, 9.5),
    "a star-forming galaxy": ("ssfr", -10, np.inf),
    "a quiescent galaxy with no ongoing star formation": ("ssfr", -np.inf, -11),
    "a distant galaxy at high redshift": (REDSHIFT, 0.5, np.inf),
    "a nearby galaxy at low redshift": (REDSHIFT, -np.inf, 0.1),
}
LADDERS = {
    "spiral": (
        "a galaxy",
        "a spiral galaxy",
        "a red spiral galaxy",
        "a red barred spiral galaxy with a ring",
    ),
    "elliptical": (
        "a galaxy",
        "an elliptical galaxy",
        "a round red elliptical galaxy",
        "a giant round red elliptical galaxy with a bright core",
    ),
    "spectrum": (
        "a galaxy spectrum",
        "a galaxy spectrum with emission lines",
        "a galaxy spectrum with strong H-alpha emission",
        "a spectrum with strong H-alpha and [OIII] emission lines at redshift 0.1",
    ),
}
KAPPA = 15
BREADTH_KAPPAS = (KAPPA, 20)
KINDS = list(range(len(VIEW_TYPES)))
TOP = (10, 100)
RESAMPLES = 1000
REPORT = Path("docs/benchmarks/text_search_quality.json")


def points(
    galaxy: np.ndarray, vectors: np.ndarray, held: np.ndarray
) -> tuple[np.ndarray, ...]:
    kept = np.isin(galaxy, held)
    return (
        np.searchsorted(held, galaxy[kept]),
        vectors[kept],
        normalize(vectors[kept] - vectors.mean(axis=0)),
    )


def cosine_tables(
    found: dict[str, tuple[np.ndarray, ...]],
    query: np.ndarray,
    direction: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw, centred = np.full((2, count, len(found)), -np.inf)
    for column, (position, vectors, centred_vectors) in enumerate(found.values()):
        raw[position, column] = vectors @ query
        centred[position, column] = centred_vectors @ direction
    return raw, centred


def best_cosine(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return table.max(axis=1), np.eye(table.shape[1])[table.argmax(axis=1)]


def likelihood(table: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    logits = kappa * table
    counts = np.isfinite(table).sum(axis=1)
    return logsumexp(logits, axis=1) - np.log(counts), softmax(logits, axis=1)


def interval(
    statistic: Callable[..., float], *samples: np.ndarray
) -> tuple[float, float]:
    low, high = bootstrap(
        samples,
        statistic,
        n_resamples=RESAMPLES,
        paired=True,
        method="percentile",
        rng=SEED,
    ).confidence_interval
    return float(low), float(high)


def precision(relevant: np.ndarray, score: np.ndarray) -> dict[str, float]:
    low, high = interval(average_precision_score, relevant, score)
    order = np.argsort(-score)
    return {
        "base_rate": round(float(relevant.mean()), 4),
        "average_precision": round(float(average_precision_score(relevant, score)), 4),
        "average_precision_low": round(float(low), 4),
        "average_precision_high": round(float(high), 4),
    } | {
        f"precision@{top}": round(float(relevant[order[:top]].mean()), 4) for top in TOP
    }


def difference(
    relevant: np.ndarray, ours: np.ndarray, theirs: np.ndarray
) -> dict[str, float]:
    def gained(relevant, ours, theirs):
        return average_precision_score(relevant, ours) - average_precision_score(
            relevant, theirs
        )

    low, high = interval(gained, relevant, ours, theirs)
    return {
        "difference": round(float(gained(relevant, ours, theirs)), 4),
        "difference_low": round(low, 4),
        "difference_high": round(high, 4),
    }


def shares(
    modes: list[str], score: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-score)
    return {
        f"{mode}@{top}": round(float(share), 4)
        for top in TOP
        for mode, share in zip(modes, weights[order[:top]].mean(axis=0))
    }


@app.function(
    image=image,
    gpu="L40S",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality() -> dict[str, Any]:
    held = np.flatnonzero(held_out(np.arange(galaxy_count())).cpu().numpy())
    space = aion_gemma_space()
    methods = {"gemma": {}, "maps": {}}
    for modality, survey in TEXT_SURVEYS.items():
        mode = f"{survey}_{modality}"
        galaxy, _, gemma = pairs(modality)
        methods["gemma"][mode] = points(galaxy, gemma.cpu().numpy(), held)
        chosen = space.modality == modality
        methods["maps"][mode] = points(
            space.galaxy[chosen], space.vectors[chosen], held
        )

    measured = provabgs(held) | {REDSHIFT: measured_redshifts(held)[0]}
    labelled = {}
    for query, (name, low, high) in LABELLED.items():
        known = ~np.isnan(measured[name])
        values = measured[name][known]
        labelled[query] = (known, (values > low) & (values < high))

    texts = list(
        dict.fromkeys(
            [*LABELLED, *(text for rungs in LADDERS.values() for text in rungs)]
        )
    )
    embedded = embed_queries(texts)
    index = text_index()
    directions = normalize(embedded - index.query_mean.numpy())
    tables = {
        text: {
            name: cosine_tables(found, query, direction, len(held))
            for name, found in methods.items()
        }
        for text, query, direction in zip(texts, embedded, directions)
    }
    rankings = {
        text: {
            name: {"raw": best_cosine(raw), "likelihood": likelihood(centred, KAPPA)}
            for name, (raw, centred) in found.items()
        }
        for text, found in tables.items()
    }
    chosen = torch.from_numpy(held)
    network, bases = stored_flow()
    joint = loaded("joint_embedding", torch.float32)
    second = indexed(generated_clouds(network, bases, joint, chosen, SEED + 1))
    cloud_breadth, seeds = {}, {}
    for text, query in zip(texts, embedded):
        scores, by_kind = likelihoods(index, query, KINDS)
        shares_by_mode = mode_shares(by_kind[chosen], KINDS)
        rankings[text]["clouds"] = {
            "likelihood": (scores[chosen].numpy(), shares_by_mode.numpy())
        }
        cloud_breadth[text] = {
            f"kappa = {kappa}": round(
                breadth(
                    likelihoods(index._replace(kappa=kappa), query, KINDS)[0][chosen]
                ),
                4,
            )
            for kappa in BREADTH_KAPPAS
        }
        if text in LABELLED:
            reseeded = likelihoods(second, query, KINDS)[0]
            agreement = spearmanr(scores[chosen], reseeded).statistic
            seeds[text] = round(float(agreement), 4)

    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": len(held),
        "kappa": KAPPA,
        "cloud_kappa": index.kappa,
        "generic_queries": len(generic_queries()),
        "queries": {
            query: {
                name: {
                    scoring: precision(relevant, score[known])
                    for scoring, (score, _) in scored.items()
                }
                for name, scored in rankings[query].items()
            }
            for query, (known, relevant) in labelled.items()
        },
        "differences": {
            query: {
                f"{name}, {scoring}": difference(
                    relevant,
                    rankings[query]["clouds"]["likelihood"][0][known],
                    score[known],
                )
                for name, scored in rankings[query].items()
                if name != "clouds"
                for scoring, (score, _) in scored.items()
            }
            for query, (known, relevant) in labelled.items()
        },
        "seed_agreement": seeds,
        "breadth": {
            ladder: {
                text: {
                    name: {
                        f"kappa = {kappa}": round(
                            breadth(torch.from_numpy(likelihood(centred, kappa)[0])), 4
                        )
                        for kappa in BREADTH_KAPPAS
                    }
                    for name, (_, centred) in tables[text].items()
                }
                | {"clouds": cloud_breadth[text]}
                for text in rungs
            }
            for ladder, rungs in LADDERS.items()
        },
        "shares": {
            text: {
                name: {
                    scoring: shares(
                        list(SHARE_NAMES) if name == "clouds" else list(methods[name]),
                        *ranked,
                    )
                    for scoring, ranked in scored.items()
                }
                for name, scored in found.items()
            }
            for text, found in rankings.items()
        },
        "memory": memory(),
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_text_search_quality.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
