"""Measures the search path against a synthetic artifact tree.

Reports artifact sizes, index load time, query latency, and the recall of the
approximate ranking against brute force over every patch. Recall is what
distinguishes a change that made the search faster from one that made it worse.

Results are written as JSON tagged with the commit and the fixture's shape, so
runs can be compared across sessions. Numbers are only comparable between trees
built with the same `--galaxies` and `--seed`.

Two caveats apply to any conclusion drawn here:

* The fixture is far smaller than production, so its index geometry differs.
  `nlist` and the `nprobe/nlist` ratio are reported for that reason: a
  fixture that probes 10% of its lists is not measuring what production, which
  probes 0.4%, does.
* Load timings run against a warm page cache on a local disk, so they are a
  lower bound on a Modal container reading a cold network volume.

    uv run python -m scripts.benchmark --galaxies 32 --out bench.json
"""

import argparse
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from app.config import ANCHOR, DIM, N_PATCHES, NPROBE, PROBE, artifact, build_dir
from app.search import Query, index, patches, search, source

PRODUCTION_GALAXIES = 17369


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

    Mirrors `search()` exactly apart from the candidate step,
    scoring against the float32 embeddings rather than the index's fp16 copies.
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
    ).astype(np.int32)
    return chosen, scores[chosen]


@dataclass
class Timing:
    """Latency of one repeated measurement, in milliseconds."""

    label: str
    samples: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.samples.append(seconds * 1000)

    def summary(self) -> dict[str, float]:
        values = np.asarray(self.samples)
        return {
            "runs": len(values),
            "p50_ms": round(float(np.percentile(values, 50)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
            "max_ms": round(float(values.max()), 3),
        }


def time_it(label: str, runs: int, call) -> Timing:
    """Run `call` `runs` times after one warm-up, recording each duration."""
    timing = Timing(label)
    call()
    for _ in range(runs):
        start = time.perf_counter()
        call()
        timing.record(time.perf_counter() - start)
    return timing


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
    scale = PRODUCTION_GALAXIES / galaxies
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

    timing = time_it("index_load", runs, load)
    built = index()
    return {
        **timing.summary(),
        "ntotal": built.ntotal,
        "nlist": built.nlist,
        "nprobe": built.nprobe,
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

            def one(batch=batch, counter=counter) -> None:
                search(batch[next(counter) % len(batch)], index=built)

            label = f"patches={patch_count},matches={matches}"
            results[label] = time_it(label, runs, one).summary()
    return results


def measure_phases(galaxies: int, runs: int, matches: int = 32) -> dict[str, Any]:
    """Where a query's time goes, split by stage of `search()`.

    Mirrors `search` step by step rather than calling it, so the stages can be
    timed individually. Keep this in step with `search` when that changes.
    """
    built = index()
    batch = queries(galaxies, runs, 4, matches)
    stages = [
        "query_direction",
        "ann_search",
        "dedup",
        "reconstruct_maps",
        "score_matmul",
    ]
    samples: dict[str, list[float]] = {stage: [] for stage in stages}
    reconstructed = 0

    for query in batch:
        marks = [time.perf_counter()]
        direction = normalize(
            built.reconstruct_batch(
                query.galaxy * N_PATCHES + np.asarray(query.patches)
            ).mean(axis=0, keepdims=True)
        )
        marks.append(time.perf_counter())
        ids = built.search(direction, PROBE)[1][0]
        marks.append(time.perf_counter())
        found, first = np.unique(ids[ids >= 0] // N_PATCHES, return_index=True)
        ranked = found[np.argsort(first)]
        order = np.concatenate(
            ([query.galaxy], ranked[ranked != query.galaxy][: query.matches])
        ).astype(np.int32)
        marks.append(time.perf_counter())
        rows = built.reconstruct_batch(
            (order[:, None] * N_PATCHES + np.arange(N_PATCHES)).reshape(-1)
        )
        marks.append(time.perf_counter())
        (rows @ direction.T).reshape(len(order), N_PATCHES)
        marks.append(time.perf_counter())

        reconstructed = rows.shape[0]
        for stage, start, end in zip(stages, marks[:-1], marks[1:], strict=True):
            samples[stage].append((end - start) * 1000)

    total = sum(float(np.median(values)) for values in samples.values())
    return {
        "matches": matches,
        "vectors_reconstructed": reconstructed,
        "total_p50_ms": round(total, 3),
        "us_per_reconstructed_vector": round(
            1000 * float(np.median(samples["reconstruct_maps"])) / max(reconstructed, 1),
            3,
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
    """Recall of the approximate ranking against brute force, swept over nprobe."""
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
        # The index is a process-wide cached singleton, shared with app.main:
        # leaving it at a swept value would silently mis-measure everything after.
        built.nprobe = original

    return {"requested_matches": matches, "queries": count, "by_nprobe": sweep}


def measure_endpoints(galaxies: int, runs: int) -> dict[str, Any]:
    """Per-request cost of the metadata endpoints, excluding the network."""
    from fastapi.testclient import TestClient

    from app.main import app

    rng = np.random.default_rng(0)
    with TestClient(app) as client:
        results = {}
        for label, path in (
            ("meta", "/meta"),
            ("tokens", "/galaxies/{}/tokens"),
            ("coverage", "/galaxies/{}/coverage"),
        ):

            def one(path=path) -> None:
                client.get(path.format(int(rng.integers(galaxies))))

            results[label] = time_it(label, runs, one).summary()
    return results


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

    from scripts.fixture import build

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

    # Read the size off the tree rather than trusting the flag: with --tree the
    # flag describes a build that did not happen, and sampling galaxy ids past
    # the end of the index would crash the search.
    galaxies = source("encoded").count_rows()
    if galaxies != arguments.galaxies:
        print(f"tree holds {galaxies} galaxies; --galaxies {arguments.galaxies} ignored")
    if galaxies < 2:
        raise SystemExit(f"need at least 2 galaxies to measure, tree holds {galaxies}")

    report = {
        "commit": commit(),
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
