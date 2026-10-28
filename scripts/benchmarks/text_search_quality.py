import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.special import logsumexp, softmax
from scipy.stats import bootstrap, entropy
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

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
KAPPA = 15
BREADTH_KAPPAS = (KAPPA, 20)
TOP = (10, 100)
RESAMPLES = 1000
REPORT = Path("docs/benchmarks/text_search_quality.json")


def generic_queries() -> list[str]:
    phrases = (
        f"{adjective} {subject}".strip()
        for adjective, subject in product(ADJECTIVES, SUBJECTS)
    )
    nouns = [("an " if phrase[0] in "aeiou" else "a ") + phrase for phrase in phrases]
    return [
        frame.format(noun + suffix)
        for frame, noun, suffix in product(FRAMES, nouns, SUFFIXES)
    ]


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


def precision(relevant: np.ndarray, score: np.ndarray) -> dict[str, float]:
    low, high = bootstrap(
        (relevant, score),
        average_precision_score,
        n_resamples=RESAMPLES,
        vectorized=False,
        paired=True,
        method="percentile",
        rng=SEED,
    ).confidence_interval
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
    modes: list[str], score: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-score)
    return {
        f"{mode}@{top}": round(float(share), 4)
        for top in TOP
        for mode, share in zip(modes, weights[order[:top]].mean(axis=0))
    }


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

    measured = pq.read_table(artifact("properties"))
    labelled = {}
    for query, (name, low, high) in LABELLED.items():
        values = measured.column(f"{name}_measured").to_numpy()[held]
        known = np.isfinite(values)
        labelled[query] = (known, (values[known] > low) & (values[known] < high))

    texts = [*LABELLED, *(text for rungs in LADDERS.values() for text in rungs)]
    embedded = embed_queries(texts)
    generic = generic_queries()
    directions = normalize(embedded - embed_queries(generic).mean(axis=0))
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

    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": len(held),
        "kappa": KAPPA,
        "generic_queries": len(generic),
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
        "breadth": {
            ladder: {
                text: {
                    name: {
                        f"kappa = {kappa}": breadth(likelihood(centred, kappa)[0])
                        for kappa in BREADTH_KAPPAS
                    }
                    for name, (_, centred) in tables[text].items()
                }
                for text in rungs
            }
            for ladder, rungs in LADDERS.items()
        },
        "shares": {
            text: {
                name: {
                    scoring: shares(list(methods[name]), *ranked)
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
