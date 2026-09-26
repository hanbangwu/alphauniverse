import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import faiss
import httpx
import modal
import numpy as np

from app.config import ARTIFACTS, DATASET_REVISION, GALAXIES, N_PATCHES
from app.cutouts import cutouts
from app.main import labels
from app.search import (
    Query,
    candidates,
    centroid,
    index,
    rank,
    score_maps,
    span_maps,
    starts,
    vectors,
)
from app.spectra import spectra
from modal_app import app, fastapi_app, serving_image

image = serving_image.add_local_python_source("modal_app")

PATCHES = 4
MATCHES = (8, 32, 128)
STAGES = ["centroid", "candidates", "vectors", "score_maps", "span_maps", "rank"]


def elapsed(call: Callable[[], Any]) -> float:
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1000


def time_it(runs: int, call: Callable[[], Any]) -> dict[str, float]:
    call()
    samples = [elapsed(call) for _ in range(runs)]
    return {
        "runs": runs,
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
    }


def queries(count: int, patch_count: int, matches: int) -> list[Query]:
    rng = np.random.default_rng(0)
    return [
        Query(
            galaxy=int(rng.integers(GALAXIES)),
            p=tuple(
                int(patch)
                for patch in rng.choice(N_PATCHES, patch_count, replace=False)
            ),
            matches=matches,
        )
        for _ in range(count)
    ]


def spec(function: modal.Function) -> dict[str, Any]:
    return {
        "cpu": function.spec.cpu,
        "memory_mb": function.spec.memory,
        "gpu": function.spec.gpus,
    }


def environment() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


@app.function(image=image, cpu=1, timeout=60 * 60)
def client(url: str, runs: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)

    def galaxy() -> int:
        return int(rng.integers(GALAXIES))

    with httpx.Client(base_url=url, timeout=None) as session:

        def get(path: str, **params: Any) -> None:
            session.get(path, params=params).raise_for_status()

        def size(role: str) -> int:
            response = session.head(f"/artifacts/{role}").raise_for_status()
            return int(response.headers["content-length"])

        def similarity(matches: int) -> None:
            get(
                "/similarity",
                galaxy=galaxy(),
                p=rng.choice(N_PATCHES, PATCHES, replace=False).tolist(),
                matches=matches,
            )

        cold_meta = round(elapsed(lambda: get("/meta")), 3)
        cold_similarity = round(elapsed(lambda: similarity(32)), 3)

        calls = {
            "meta": lambda: get("/meta"),
            "image": lambda: get(f"/galaxies/{galaxy()}/image.png"),
            "tokens": lambda: get(f"/galaxies/{galaxy()}/tokens"),
            "coverage": lambda: get(f"/galaxies/{galaxy()}/coverage"),
        } | {
            f"similarity matches={matches}": (
                lambda matches=matches: similarity(matches)
            )
            for matches in MATCHES
        }
        return {
            "cold_meta_ms": cold_meta,
            "cold_similarity_ms": cold_similarity,
            "warm": {label: time_it(runs, call) for label, call in calls.items()},
            "artifact_bytes": {role: size(role) for role in ARTIFACTS},
        }


def stage_times(query: Query, built: faiss.Index) -> tuple[list[float], int]:
    marks = [time.perf_counter()]
    direction = centroid(query, index=built)
    marks.append(time.perf_counter())
    order = candidates(query, direction, index=built)
    marks.append(time.perf_counter())
    rows = vectors(order, index=built)
    marks.append(time.perf_counter())
    scored = score_maps(rows, direction, width=N_PATCHES)
    marks.append(time.perf_counter())
    spectral_scores = span_maps(order, direction, index=built)
    marks.append(time.perf_counter())
    rank(order, scored, spectral_scores)
    marks.append(time.perf_counter())
    times = [(end - begin) * 1000 for begin, end in pairwise(marks)]
    return times, len(rows)


@app.function(
    image=image,
    cpu=fastapi_app.spec.cpu,
    memory=fastapi_app.spec.memory,
    volumes=fastapi_app.spec.volumes,
    timeout=60 * 60,
)
def stages(runs: int, matches: int = 32) -> dict[str, Any]:
    loads = {
        load.__name__: round(elapsed(load), 3)
        for load in (labels, cutouts, spectra, index, starts)
    }
    built = index()

    batch = queries(runs + 1, PATCHES, matches)
    cold = stage_times(batch[0], built)[0]
    samples: dict[str, list[float]] = {name: [] for name in STAGES}
    widths: list[int] = []

    for query in batch[1:]:
        times, width = stage_times(query, built)
        widths.append(width)
        for name, milliseconds in zip(STAGES, times, strict=True):
            samples[name].append(milliseconds)

    total = sum(float(np.median(values)) for values in samples.values())
    reconstructed = int(np.median(widths))
    return {
        "environment": environment(),
        "load_ms": loads,
        "runs": runs,
        "matches": matches,
        "vectors_reconstructed": reconstructed,
        "cold_stages_ms": {
            name: round(milliseconds, 3)
            for name, milliseconds in zip(STAGES, cold, strict=True)
        },
        "total_p50_ms": round(total, 3),
        "us_per_reconstructed_vector": round(
            1000 * float(np.median(samples["vectors"])) / reconstructed, 3
        ),
        "stages": {
            name: {
                "p50_ms": round(float(np.median(values)), 3),
                "share": round(float(np.median(values)) / total, 4),
            }
            for name, values in samples.items()
        },
    }


@app.local_entrypoint()
def main(runs: int = 30) -> None:
    report = {
        "commit": subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], text=True
        ).strip(),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": spec(fastapi_app),
        "client": spec(client),
        "requests": client.remote(fastapi_app.get_web_url(), runs),
        "stages": stages.remote(runs),
    }
    print(json.dumps(report, indent=2))
