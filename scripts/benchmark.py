import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any, get_args

import faiss
import httpx
import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from app.config import (
    DATASET_REVISION,
    N_PATCHES,
    N_SPANS,
    SPECTRUM_ORIGIN,
    SPECTRUM_TOKEN_WIDTH,
    Download,
    Projection,
    galaxy_count,
)
from app.dataset import dataset, labels
from app.main import SPECTRUM_SURVEY
from app.search import (
    Query,
    candidates,
    centroid,
    index,
    rank,
    score_maps,
    search,
    span_maps,
    starts,
    vectors,
)
from modal_app import (
    SERVING_CPU,
    SERVING_MAX_CONTAINERS,
    SERVING_MAX_INPUTS,
    SERVING_MEMORY,
    SERVING_SCALEDOWN_WINDOW,
    app,
    fastapi_app,
    serving_image,
)

image = serving_image.add_local_python_source("modal_app")
logger = logging.getLogger(__name__)

PATCHES = 4
SPANS = 4
MATCHES = (8, 32, 128)
CLIENTS = (1, 4, SERVING_MAX_INPUTS, 2 * SERVING_MAX_INPUTS)
STAGES = ["centroid", "candidates", "vectors", "score_maps", "span_maps", "rank"]
SOURCES = ["app", "scripts", "modal_app.py"]
REPORT = Path("docs/benchmarks/latest.json")
ENTRY = (
    "import json, sys; from app.config import DATASET_REVISION; "
    "from scripts.benchmark import stages; "
    "print(json.dumps(stages.local(int(sys.argv[1])) | {'revision': DATASET_REVISION}))"
)


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


def time_it(runs: int, call: Callable[[], Any]) -> dict[str, Any]:
    call()
    return summary([elapsed(call) for _ in range(runs)])


def timed_requests(session: httpx.Client, batch: list[dict[str, Any]]) -> list[float]:
    def ask(params: dict[str, Any]) -> None:
        session.get("/search", params=params).raise_for_status()

    return [elapsed(partial(ask, params)) for params in batch]


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
    }


@app.function(image=image, cpu=1, timeout=60 * 60)
def client(url: str, runs: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)

    with httpx.Client(base_url=url, timeout=None) as session:

        def get(path: str, **params: Any) -> None:
            session.get(path, params=params).raise_for_status()

        def size(path: str) -> int:
            response = session.head(path).raise_for_status()
            return int(response.headers["content-length"])

        def parameters(matches: int) -> dict[str, Any]:
            return {
                "galaxy": galaxy(),
                "p": rng.choice(N_PATCHES, PATCHES, replace=False).tolist(),
                "matches": matches,
            }

        def similarity(matches: int) -> None:
            get("/search", **parameters(matches))

        cold_meta = round(elapsed(lambda: get("/meta")), 3)
        galaxies = session.get("/meta").raise_for_status().json()["galaxies"]

        def galaxy() -> int:
            return int(rng.integers(galaxies))

        cold_similarity = round(elapsed(lambda: similarity(32)), 3)

        calls = {
            "meta": lambda: get("/meta"),
            "image": lambda: get(f"/galaxy/{galaxy()}/image"),
            "tokens": lambda: get(f"/galaxy/{galaxy()}/image/tokens"),
            "coverage": lambda: get(f"/galaxy/{galaxy()}"),
        } | {
            f"similarity matches={matches}": (
                lambda matches=matches: similarity(matches)
            )
            for matches in MATCHES
        }
        warm = {label: time_it(runs, call) for label, call in calls.items()}

        def concurrent(clients: int) -> dict[str, Any]:
            batches = [
                [parameters(32) for _ in range(runs + 1)] for _ in range(clients)
            ]
            with ExitStack() as stack, ThreadPoolExecutor(clients) as pool:
                sessions = [
                    stack.enter_context(httpx.Client(base_url=url, timeout=None))
                    for _ in batches
                ]
                list(
                    pool.map(timed_requests, sessions, [batch[:1] for batch in batches])
                )
                begin = time.perf_counter()
                timed = list(
                    pool.map(timed_requests, sessions, [batch[1:] for batch in batches])
                )
                seconds = time.perf_counter() - begin
            samples = [sample for client_samples in timed for sample in client_samples]
            return summary(samples) | {
                "requests_per_s": round(len(samples) / seconds, 2)
            }

        cells = pq.read_table(
            io.BytesIO(session.get("/downloads/tokens").raise_for_status().content),
            columns=[SPECTRUM_SURVEY],
        )
        holders = np.flatnonzero(
            pc.is_valid(cells.column(SPECTRUM_SURVEY)).to_numpy(zero_copy_only=False)
        )

        def spectrum(route: str) -> None:
            get(f"/galaxy/{rng.choice(holders)}/spectrum{route}")

        def span_query(patch_count: int) -> dict[str, Any]:
            galaxy = int(rng.choice(holders))
            response = session.get(f"/galaxy/{galaxy}/spectrum")
            table = pa.ipc.open_stream(response.raise_for_status().content).read_all()
            spans = observed_spans(table.column("wavelength").to_numpy())
            return {
                "galaxy": galaxy,
                "p": rng.choice(N_PATCHES, patch_count, replace=False).tolist(),
                "s": rng.choice(spans, SPANS, replace=False).tolist(),
                "matches": 32,
            }

        def span_similarity(patch_count: int) -> Callable[[], None]:
            batch = iter([span_query(patch_count) for _ in range(runs + 1)])
            return lambda: get("/search", **next(batch))

        spectral_calls = {
            name: lambda route=route: spectrum(route)
            for name, route in (("spectrum", ""), ("spectrum tokens", "/tokens"))
        } | {
            "similarity spans matches=32": span_similarity(0),
            "similarity patches and spans matches=32": span_similarity(PATCHES),
        }
        return {
            "environment": environment(),
            "cold_meta_ms": cold_meta,
            "cold_similarity_ms": cold_similarity,
            "warm": warm
            | {label: time_it(runs, call) for label, call in spectral_calls.items()},
            "concurrency": {
                f"clients={clients}": concurrent(clients) for clients in CLIENTS
            },
            "artifact_bytes": {
                f"{projection}_points": size(f"/projections/{projection}")
                for projection in get_args(Projection)
            }
            | {role: size(f"/downloads/{role}") for role in get_args(Download)},
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


def whole_searches(runs: int, matches: int, built: faiss.Index) -> dict[str, Any]:
    batch = queries(runs + 1, PATCHES, matches)
    search(batch[0], index=built)
    samples, searches = [], []
    for query in batch[1:]:
        faiss.cvar.indexIVF_stats.reset()
        samples.append(elapsed(partial(search, query, index=built)))
        searches.append(faiss.cvar.indexIVF_stats.nq)
    return summary(samples) | {
        "looked_further": round(float(np.mean(np.greater(searches, 1))), 4),
        "most_searches": max(searches),
    }


@app.function(image=image)
def stages(runs: int, matches: int = 32) -> dict[str, Any]:
    loads = {
        load.__name__: round(elapsed(load), 3)
        for load in (dataset, labels, index, starts)
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
        "searches": {
            f"matches={asked}": whole_searches(runs, asked, built) for asked in MATCHES
        },
    }


@app.function(
    image=image,
    cpu=SERVING_CPU,
    memory=SERVING_MEMORY,
    volumes=fastapi_app.spec.volumes,
    timeout=6 * 60 * 60,
)
def paired(sources: dict[str, bytes], order: list[str], runs: int) -> dict[str, Any]:
    report = {"environment": environment(), "order": order}
    root = Path(tempfile.mkdtemp())
    for name, source in sources.items():
        with tarfile.open(fileobj=io.BytesIO(source)) as bundle:
            bundle.extractall(root / name, filter="data")

    rounds: dict[str, list[dict[str, Any]]] = {name: [] for name in sources}
    for position, name in enumerate(order):
        try:
            completed = subprocess.run(
                [sys.executable, "-c", ENTRY, str(runs)],
                cwd=root / name,
                capture_output=True,
                text=True,
                check=True,
                timeout=60 * 60,
            )
            result = json.loads(completed.stdout.splitlines()[-1])
        except subprocess.CalledProcessError as failure:
            result = {"error": f"exit {failure.returncode}: {failure.stderr[-2000:]}"}
        except subprocess.TimeoutExpired:
            result = {"error": "no result within an hour"}
        except (ValueError, IndexError):
            result = {"error": f"no report on stdout: {completed.stdout[-2000:]}"}
        rounds[name].append(result | {"position": position})
    return report | {"rounds": rounds}


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def attempt(part: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except Exception as failure:
        logger.exception("the %s part of the benchmark failed", part)
        return {"error": f"{type(failure).__name__}: {failure}"}


@app.local_entrypoint()
def main(runs: int = 30) -> None:
    if git("status", "--porcelain"):
        raise SystemExit("Commit every change first: the benchmark times commits.")
    git("fetch", "origin", "main")
    commits = {
        "after": git("rev-parse", "HEAD"),
        "before": git("rev-parse", "origin/main"),
    }
    code = {
        git("rev-parse", *(f"{commit}:{source}" for source in SOURCES))
        for commit in commits.values()
    }
    notes = []
    stored = json.loads(REPORT.read_text()).get("best") if REPORT.exists() else None
    if stored:
        try:
            stored_code = git(
                "rev-parse", *(f"{stored['commit']}:{source}" for source in SOURCES)
            )
        except subprocess.CalledProcessError:
            notes.append(f"best {stored['commit']} is not in this clone")
            stored = None
        else:
            if stored_code not in code:
                commits["best"] = stored["commit"]
    names = list(commits)
    sources = {
        name: subprocess.check_output(["git", "archive", commit, *SOURCES])
        for name, commit in commits.items()
    }
    report = {
        "commit": git("describe", "--always"),
        "commits": commits,
        "notes": notes,
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": {
            "cpu": SERVING_CPU,
            "memory_mb": SERVING_MEMORY,
            "max_inputs": SERVING_MAX_INPUTS,
            "max_containers": SERVING_MAX_CONTAINERS,
            "scaledown_window_s": SERVING_SCALEDOWN_WINDOW,
        },
        "client": spec(client),
        "requests": attempt(
            "requests", lambda: client.remote(fastapi_app.get_web_url(), runs)
        ),
        "stages": attempt(
            "stages",
            lambda: (
                paired.remote(sources, names + names[::-1], runs)
                | {"locked_dependencies": "after"}
            ),
        ),
    }
    means = {
        name: float(np.mean([entry["total_p50_ms"] for entry in rounds]))
        for name, rounds in report["stages"].get("rounds", {}).items()
        if all("error" not in entry for entry in rounds)
    }
    if means:
        best = min(means, key=means.__getitem__)
        report["best"] = {
            "name": best,
            "commit": commits[best],
            "total_p50_ms": round(means[best], 3),
        }
    elif stored:
        report["best"] = {"commit": stored["commit"]}
        notes.append("no version timed fully, so best keeps the stored commit")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
