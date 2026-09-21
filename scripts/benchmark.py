"""Measures the search path against a synthetic artifact tree.

Reports artifact sizes, index load time, query latency, and the recall of the
approximate ranking against brute force over every patch. Recall is what
distinguishes a change that made the search faster from one that made it worse.
`docs/testing.md` says how to run it and `docs/performance.md` how to read it.
"""

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi.testclient import TestClient
from sklearn.preprocessing import normalize

from app.config import (
    ANCHOR,
    DIM,
    GALAXIES,
    N_PATCHES,
    NPROBE,
    PROBE,
    artifact,
    build_dir,
)
from app.main import app
from app.search import (
    Query,
    candidates,
    centroid,
    index,
    patches,
    rank,
    score_maps,
    search,
    source,
    vectors,
)
from scripts.fixture import build


def anchor_patches() -> np.ndarray:
    """Every galaxy's anchor patches, normalised: the exact search's corpus.

    Holds the whole corpus in memory, which is only viable at fixture scale.
    """
    cell = source("encoded").to_table(columns=[ANCHOR]).column(ANCHOR).combine_chunks()
    return patches(cell)


def exact_ranking(
    query: Query, corpus: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force ranking over every patch: the ground truth for recall.

    Mirrors `search()` apart from the candidate step, scoring against the
    float32 embeddings rather than the index's fp16 copies.
    """
    rows = anchor_patches() if corpus is None else corpus
    direction = normalize(
        rows[query.galaxy * N_PATCHES + np.asarray(query.patches)].mean(
            axis=0, keepdims=True
        )
    )
    scores = (rows @ direction.T).reshape(-1, N_PATCHES).max(axis=1)
    order = np.argsort(-scores, kind="stable")
    chosen = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return chosen, scores[chosen]


def time_it(runs: int, call) -> dict[str, float]:
    """Time `call` over `runs` runs after one warm-up, in milliseconds."""
    call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    values = np.asarray(samples)
    return {
        "runs": runs,
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(float(values.max()), 3),
    }


def queries(galaxies: int, count: int, patch_count: int, matches: int) -> list[Query]:
    """A deterministic spread of queries across the tree."""
    rng = np.random.default_rng(0)
    return [
        Query(
            galaxy=int(rng.integers(galaxies)),
            p=tuple(
                int(patch)
                for patch in rng.choice(N_PATCHES, patch_count, replace=False)
            ),
            matches=matches,
        )
        for _ in range(count)
    ]


def measure_sizes(galaxies: int) -> dict[str, Any]:
    """Artifact sizes, with a linear extrapolation to production scale."""
    scale = GALAXIES / galaxies
    files = {}
    for path in sorted(build_dir().iterdir()):
        size = path.stat().st_size
        files[path.name] = {
            "bytes": size,
            "mb": round(size / 1e6, 2),
            "projected_gb_at_production": round(size * scale / 1e9, 2),
        }
    return files


def measure_load(runs: int) -> dict[str, Any]:
    """Cost of building the in-process index handle from disk."""

    def load() -> None:
        index.cache_clear()
        index()

    timing = time_it(runs, load)
    built = index()
    return {
        **timing,
        "ntotal": built.ntotal,
        "nlist": built.nlist,
        "nprobe": built.nprobe,
        "probe": PROBE,
        "probe_fraction": round(built.nprobe / built.nlist, 4),
        "index_bytes": artifact("encoded_index").stat().st_size,
    }


def measure_latency(galaxies: int, runs: int) -> dict[str, Any]:
    """Query latency across query shapes, against the warm index."""
    built = index()
    results = {}
    for patch_count in (1, 4, 16):
        for matches in (8, 32, 128):
            batch = queries(galaxies, runs, patch_count, matches)
            counter = itertools.count()

            def one() -> None:
                search(batch[next(counter) % len(batch)], index=built)

            results[f"patches={patch_count},matches={matches}"] = time_it(runs, one)
    return results


def measure_phases(galaxies: int, runs: int, matches: int = 32) -> dict[str, Any]:
    """Where a query's time goes, split by stage of `search()`."""
    built = index()
    batch = queries(galaxies, runs, 4, matches)
    stages = ["centroid", "candidates", "vectors", "score_maps", "rank"]
    samples: dict[str, list[float]] = {stage: [] for stage in stages}
    widths: list[int] = []

    for query in batch:
        marks = [time.perf_counter()]
        direction = centroid(query, index=built)
        marks.append(time.perf_counter())
        order = candidates(query, direction, index=built)
        marks.append(time.perf_counter())
        rows = vectors(order, index=built)
        marks.append(time.perf_counter())
        scored = score_maps(rows, direction)
        marks.append(time.perf_counter())
        rank(order, scored)
        marks.append(time.perf_counter())

        widths.append(rows.shape[0])
        for stage, start, end in zip(stages, marks[:-1], marks[1:], strict=True):
            samples[stage].append((end - start) * 1000)

    total = sum(float(np.median(values)) for values in samples.values())
    reconstructed = int(np.median(widths))
    return {
        "matches": matches,
        "vectors_reconstructed": reconstructed,
        "total_p50_ms": round(total, 3),
        "us_per_reconstructed_vector": round(
            1000 * float(np.median(samples["vectors"])) / reconstructed, 3
        ),
        "stages": {
            stage: {
                "p50_ms": round(float(np.median(values)), 3),
                "share": round(float(np.median(values)) / total, 4),
            }
            for stage, values in samples.items()
        },
    }


def measure_recall(galaxies: int, count: int, matches: int) -> dict[str, Any]:
    """Recall against brute force, swept over `nprobe`."""
    corpus = anchor_patches()
    built = index()
    original = built.nprobe

    batch = queries(galaxies, count, 4, matches)
    truth = [set(exact_ranking(query, corpus)[0][1:].tolist()) for query in batch]

    sweep = {}
    try:
        for nprobe in sorted({1, 4, 16, NPROBE, built.nlist}):
            if nprobe > built.nlist:
                continue
            built.nprobe = nprobe
            recalls, returned, elapsed = [], [], []
            for query, expected in zip(batch, truth, strict=True):
                start = time.perf_counter()
                found, _, _ = search(query, index=built)
                elapsed.append((time.perf_counter() - start) * 1000)
                got = set(found[1:].tolist())
                returned.append(len(got))
                recalls.append(len(got & expected) / max(len(expected), 1))
            sweep[str(nprobe)] = {
                "recall": round(float(np.mean(recalls)), 4),
                "returned_mean": round(float(np.mean(returned)), 2),
                "requested": matches,
                "p50_ms": round(float(np.percentile(elapsed, 50)), 3),
            }
    finally:
        built.nprobe = original

    return {"requested_matches": matches, "queries": count, "by_nprobe": sweep}


def measure_endpoints(galaxies: int, runs: int) -> dict[str, Any]:
    """Per-request cost of the endpoints that are not `/similarity`."""
    rng = np.random.default_rng(0)
    with TestClient(app) as client:
        results = {}
        for label, path in (
            ("meta", "/meta"),
            ("tokens", "/galaxies/{}/tokens"),
            ("coverage", "/galaxies/{}/coverage"),
            ("image", "/galaxies/{}/image.png"),
        ):

            def one() -> None:
                client.get(path.format(int(rng.integers(galaxies))))

            results[label] = time_it(runs, one)
    return results


def environment() -> dict[str, Any]:
    """Thread configuration, which the stage shares depend on.

    `vectors` is faiss-parallel and the `score_maps` GEMV contends with that
    pool, so two runs only compare when these agree.
    """
    return {
        "cpu_count": os.cpu_count(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galaxies",
        type=int,
        default=32,
        help="galaxies to build; ignored when --tree supplies an existing one",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--recall-queries", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="reuse an existing fixture tree instead of building one",
    )
    arguments = parser.parse_args()

    if arguments.tree:
        os.environ["ALPHAUNIVERSE_CACHE"] = str(arguments.tree.resolve())
    else:
        root = Path(".cache") / "benchmark"
        os.environ["ALPHAUNIVERSE_CACHE"] = str(root.resolve())
        source.cache_clear()
        index.cache_clear()
        started = time.perf_counter()
        build(arguments.galaxies, arguments.seed)
        print(f"built fixture in {time.perf_counter() - started:.1f}s")

    source.cache_clear()
    index.cache_clear()

    galaxies = source("encoded").count_rows()
    if galaxies != arguments.galaxies:
        print(
            f"tree holds {galaxies} galaxies; --galaxies {arguments.galaxies} ignored"
        )
    if galaxies < 2:
        raise SystemExit(f"need at least 2 galaxies to measure, tree holds {galaxies}")

    report = {
        "commit": commit(),
        "environment": environment(),
        "fixture": {
            "galaxies": galaxies,
            "seed": arguments.seed,
            "patches_per_galaxy": N_PATCHES,
            "dim": DIM,
        },
        "sizes": measure_sizes(galaxies),
        "index_load": measure_load(min(arguments.runs, 5)),
        "search": measure_latency(galaxies, arguments.runs),
        "phases": measure_phases(galaxies, min(arguments.runs, 20)),
        "recall": measure_recall(
            galaxies, arguments.recall_queries, min(32, galaxies - 1)
        ),
        "endpoints": measure_endpoints(galaxies, arguments.runs),
    }

    print(json.dumps(report, indent=2))
    if arguments.out:
        arguments.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwritten to {arguments.out}")


if __name__ == "__main__":
    main()
