import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import modal
import numpy as np
from threadpoolctl import threadpool_info

from app.config import (
    N_PATCHES,
    N_SPANS,
    SPECTRUM_ORIGIN,
    SPECTRUM_TOKEN_WIDTH,
    galaxy_count,
)
from app.search import Query
from modal_app import (
    SERVING_CPU,
    SERVING_MAX_CONTAINERS,
    SERVING_MAX_INPUTS,
    SERVING_MEMORY,
    SERVING_SCALEDOWN_WINDOW,
)

PATCHES = 4
SPANS = 4
MATCHES = (8, 32, 128)


def elapsed(call: Callable[[], Any]) -> float:
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1000


def summary(samples: list[float]) -> dict[str, Any]:
    return {
        "runs": len(samples),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
    }


def queries(count: int, patch_count: int, matches: int) -> list[Query]:
    rng = np.random.default_rng(0)
    return [
        Query(
            galaxy=int(rng.integers(galaxy_count())),
            p=tuple(
                int(patch)
                for patch in rng.choice(N_PATCHES, patch_count, replace=False)
            ),
            matches=matches,
        )
        for _ in range(count)
    ]


def observed_spans(wavelength: np.ndarray) -> np.ndarray:
    first, last = np.floor(
        (np.array([wavelength.min(), wavelength.max()]) - SPECTRUM_ORIGIN)
        / SPECTRUM_TOKEN_WIDTH
    ).astype(int)
    return np.arange(max(first, 0), min(last, N_SPANS - 1) + 1)


def spec(function: modal.Function) -> dict[str, Any]:
    return {
        "cpu": function.spec.cpu,
        "memory_mb": function.spec.memory,
        "gpu": function.spec.gpus,
    }


def server() -> dict[str, Any]:
    return {
        "cpu": SERVING_CPU,
        "memory_mb": SERVING_MEMORY,
        "max_inputs": SERVING_MAX_INPUTS,
        "max_containers": SERVING_MAX_CONTAINERS,
        "scaledown_window_s": SERVING_SCALEDOWN_WINDOW,
    }


def environment() -> dict[str, Any]:
    processor = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    fields = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in processor.splitlines())
    }
    return {
        "cpu": {
            key: fields.get(key)
            for key in ("vendor_id", "cpu family", "model", "model name")
        },
        "cpu_count": os.cpu_count(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "thread_pools": threadpool_info(),
    }


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()
