import os
import resource
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
from threadpoolctl import threadpool_info

from app.config import (
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    SPECTRUM_ORIGIN,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_WIDTH,
    galaxy_count,
)
from app.dataset import spectrum
from app.search import Query
from modal_app import (
    SERVING_CPU,
    SERVING_MAX_CONTAINERS,
    SERVING_MAX_INPUTS,
    SERVING_MEMORY,
    SERVING_SCALEDOWN_WINDOW,
)

IMAGE_TOKENS = 4
SPECTRUM_TOKENS = 4
TABLE_VALUES = 4
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


def queries(count: int, image_token_count: int, matches: int) -> list[Query]:
    rng = np.random.default_rng(0)
    return [
        Query(
            galaxy=int(rng.integers(galaxy_count())),
            p=tuple(
                int(image_token)
                for image_token in rng.choice(
                    N_IMAGE_TOKENS, image_token_count, replace=False
                )
            ),
            matches=matches,
        )
        for _ in range(count)
    ]


def wavelength(galaxy: int) -> np.ndarray:
    tables = (spectrum(galaxy, survey) for survey in SPECTRUM_SURVEYS)
    found = next(table for table in tables if table is not None)
    return found.column("wavelength").to_numpy()


def observed_spectrum_tokens(wavelength: np.ndarray) -> np.ndarray:
    first, last = np.floor(
        (np.array([wavelength.min(), wavelength.max()]) - SPECTRUM_ORIGIN)
        / SPECTRUM_TOKEN_WIDTH
    ).astype(int)
    return np.arange(max(first, 0), min(last, N_SPECTRUM_TOKENS - 1) + 1)


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


def memory() -> dict[str, float]:
    resident = {"files": 0, "anonymous": 0}
    kind = "anonymous"
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "Rss:":
                resident[kind] += int(fields[1])
            elif not fields[0].endswith(":"):
                file_backed = len(fields) > 5 and os.path.isfile(fields[5])
                kind = "files" if file_backed else "anonymous"
    return {
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "arrow_peak_mib": round(pa.default_memory_pool().max_memory() / 2**20, 1),
        "resident_files_mib": round(resident["files"] / 1024, 1),
        "resident_anonymous_mib": round(resident["anonymous"] / 1024, 1),
    }


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()
