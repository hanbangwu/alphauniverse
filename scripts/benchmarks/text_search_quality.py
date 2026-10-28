import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import average_precision_score

from app.alignment import held_out, pairs
from app.config import (
    DATASET_REVISION,
    REDSHIFT,
    SEED,
    TEXT_SURVEYS,
    artifact,
    galaxy_count,
)
from app.text_search import aion_gemma_space, embed_queries
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
FRAMES = ("{}", "an image of {}", "the spectrum of {}", "the photometry of {}")
ADJECTIVES = (
    "",
    "bright",
    "faint",
    "red",
    "blue",
    "massive",
    "small",
    "large",
    "nearby",
    "distant",
    "dusty",
    "star-forming",
    "quiescent",
    "compact",
)
SUBJECTS = (
    "galaxy",
    "spiral galaxy",
    "elliptical galaxy",
    "disk galaxy",
    "irregular galaxy",
    "dwarf galaxy",
    "lenticular galaxy",
    "barred spiral galaxy",
    "pair of merging galaxies",
    "compact galaxy",
    "edge-on galaxy",
    "face-on spiral galaxy",
    "starburst galaxy",
    "quasar host galaxy",
    "cluster galaxy",
    "low surface brightness galaxy",
)
SUFFIXES = (
    "",
    " with spiral arms",
    " with a bright bulge",
    " with a central bar",
    " with tidal tails",
    " with strong emission lines",
    " with an old stellar population",
    " at high redshift",
    " at low redshift",
    " with a dust lane",
    " with a close companion",
    " with an active nucleus",
)
KAPPAS = (15, 20)
TOP = (10, 100)
RESAMPLES = 1000
REPORT = Path("docs/benchmarks/text_search_quality.json")


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def generic_queries() -> list[str]:
    phrases = (
        f"{adjective} {subject}".strip()
        for adjective, subject in product(ADJECTIVES, SUBJECTS)
    )
    nouns = [("an " if phrase[0] in "aeiou" else "a ") + phrase for phrase in phrases]
    return list(
        dict.fromkeys(
            frame.format(noun + suffix)
            for frame, noun, suffix in product(FRAMES, nouns, SUFFIXES)
        )
    )


def points(
    rows: dict[str, tuple[np.ndarray, np.ndarray]], held: np.ndarray
) -> dict[str, np.ndarray]:
    parts = []
    for mode, (galaxy, vectors) in enumerate(rows.values()):
        kept = np.isin(galaxy, held)
        parts.append(
            (
                np.searchsorted(held, galaxy[kept]),
                np.full(kept.sum(), mode),
                vectors[kept],
                unit(vectors[kept] - vectors.mean(axis=0)),
            )
        )
    owner, mode, raw, centred = (np.concatenate(column) for column in zip(*parts))
    order = np.argsort(owner, kind="stable")
    return {
        "owner": owner[order],
        "mode": mode[order],
        "raw": raw[order],
        "centred": centred[order],
        "starts": np.flatnonzero(np.diff(owner[order], prepend=-1)),
    }


def best_cosine(
    found: dict[str, np.ndarray], query: np.ndarray
) -> tuple[np.ndarray, ...]:
    cosine = found["raw"] @ query
    score = np.maximum.reduceat(cosine, found["starts"])
    chosen = (cosine == score[found["owner"]]).astype(np.float64)
    return score, chosen / np.add.reduceat(chosen, found["starts"])[found["owner"]]


def likelihood(
    found: dict[str, np.ndarray], direction: np.ndarray, kappa: float
) -> tuple[np.ndarray, ...]:
    logits = kappa * (found["centred"] @ direction)
    peak = np.maximum.reduceat(logits, found["starts"])
    weights = np.exp(logits - peak[found["owner"]])
    total = np.add.reduceat(weights, found["starts"])
    counts = np.diff(found["starts"], append=len(logits))
    return peak + np.log(total / counts), weights / total[found["owner"]]


def precision(
    relevant: np.ndarray, score: np.ndarray, draws: np.ndarray
) -> dict[str, float]:
    resampled = [average_precision_score(relevant[draw], score[draw]) for draw in draws]
    low, high = np.percentile(resampled, [2.5, 97.5])
    order = np.argsort(-score)
    return {
        "base_rate": round(float(relevant.mean()), 4),
        "average_precision": round(float(average_precision_score(relevant, score)), 4),
        "average_precision_low": round(float(low), 4),
        "average_precision_high": round(float(high), 4),
    } | {
        f"precision@{top}": round(float(relevant[order[:top]].mean()), 4) for top in TOP
    }


def shares(
    found: dict[str, np.ndarray], score: np.ndarray, weights: np.ndarray
) -> dict[str, dict[str, float]]:
    order = np.argsort(-score)
    summed = {}
    for top in TOP:
        members = np.isin(found["owner"], order[:top])
        totals = np.bincount(
            found["mode"][members], weights[members], minlength=len(TEXT_SURVEYS)
        )
        summed[f"top{top}"] = dict(zip(TEXT_SURVEYS, (totals / top).round(4).tolist()))
    return summed


def breadth(score: np.ndarray) -> float:
    return round(float(np.exp(entropy(softmax(score))) / len(score)), 4)


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality() -> dict[str, Any]:
    held = np.flatnonzero(held_out(np.arange(galaxy_count())).cpu().numpy())
    observed = {}
    for modality in TEXT_SURVEYS:
        galaxy, _, gemma = pairs(modality)
        observed[modality] = (galaxy, gemma.cpu().numpy())
    space = aion_gemma_space()
    methods = {
        "gemma": points(observed, held),
        "maps": points(
            {
                modality: (
                    space.galaxy[space.modality == modality],
                    space.vectors[space.modality == modality],
                )
                for modality in TEXT_SURVEYS
            },
            held,
        ),
    }

    measured = pq.read_table(
        artifact("properties"),
        columns=[f"{name}_measured" for name in ("mass", "ssfr", REDSHIFT)],
    )
    labelled = {}
    for query, (name, low, high) in LABELLED.items():
        values = measured.column(f"{name}_measured").to_numpy()[held]
        known = np.flatnonzero(np.isfinite(values))
        draws = np.random.default_rng(SEED).integers(
            len(known), size=(RESAMPLES, len(known))
        )
        labelled[query] = (known, (values[known] > low) & (values[known] < high), draws)

    texts = list(
        dict.fromkeys(
            [*LABELLED, *(text for rungs in LADDERS.values() for text in rungs)]
        )
    )
    queries = dict(zip(texts, embed_queries(texts)))
    generic = generic_queries()
    bank = embed_queries(generic).mean(axis=0)
    directions = {text: unit(query - bank) for text, query in queries.items()}
    rankings = {
        (text, name, scoring): ranked
        for text in texts
        for name, found in methods.items()
        for scoring, ranked in (
            ("raw", best_cosine(found, queries[text])),
            ("likelihood", likelihood(found, directions[text], KAPPAS[0])),
        )
    }

    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": len(held),
        "kappa": KAPPAS[0],
        "generic_queries": len(generic),
        "queries": {
            query: {
                name: {
                    scoring: precision(
                        relevant, rankings[query, name, scoring][0][known], draws
                    )
                    for scoring in ("raw", "likelihood")
                }
                for name in methods
            }
            for query, (known, relevant, draws) in labelled.items()
        },
        "breadth": {
            ladder: {
                text: {
                    name: {
                        str(kappa): breadth(
                            likelihood(found, directions[text], kappa)[0]
                        )
                        for kappa in KAPPAS
                    }
                    for name, found in methods.items()
                }
                for text in rungs
            }
            for ladder, rungs in LADDERS.items()
        },
        "shares": {
            text: {
                name: {
                    scoring: shares(found, *rankings[text, name, scoring])
                    for scoring in ("raw", "likelihood")
                }
                for name, found in methods.items()
            }
            for text in texts
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
