import io
import json
import resource
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import DATASET_REVISION, N_IMAGE_TOKENS, artifact, galaxy_count, labels
from app.search import (
    Query,
    candidates,
    centroid,
    index,
    rank,
    scalar_maps,
    score_maps,
    search,
    span_maps,
    starts,
    tokens,
    vectors,
)
from modal_app import (
    CACHE_PATH,
    SERVING_CPU,
    SERVING_MEMORY,
    app,
    cache_volume,
    serving_image,
)
from scripts.benchmarks.common import (
    IMAGE_TOKENS,
    MATCHES,
    elapsed,
    environment,
    git,
    memory,
    queries,
    server,
    summary,
)

image = serving_image.add_local_python_source("modal_app")

STAGES = [
    "centroid",
    "candidates",
    "vectors",
    "score_maps",
    "span_maps",
    "scalar_maps",
    "rank",
]
KINDS = ("wall", "user", "system")
SOURCES = ["app", "scripts", "modal_app.py"]
REPORT = Path("docs/benchmarks/search_performance.json")
ENTRY = (
    "import json, sys, time\n"
    "start = time.perf_counter()\n"
    "import app.main\n"
    "import_ms = round((time.perf_counter() - start) * 1000, 3)\n"
    "from app.config import DATASET_REVISION\n"
    "try:\n"
    "    from scripts.benchmarks.search_performance import stages\n"
    "except ModuleNotFoundError:\n"
    "    from scripts.benchmark import stages\n"
    "print(json.dumps(stages.local(int(sys.argv[1]))"
    " | {'revision': DATASET_REVISION, 'import_ms': import_ms}))"
)


def mark() -> tuple[float, float, float]:
    used = resource.getrusage(resource.RUSAGE_SELF)
    return time.perf_counter(), used.ru_utime, used.ru_stime


def usages(
    names: list[str], marks: list[tuple[float, float, float]]
) -> dict[str, dict[str, float]]:
    return {
        name: {
            kind: (end - begin) * 1000
            for kind, begin, end in zip(KINDS, start, stop, strict=True)
        }
        for name, (start, stop) in zip(names, pairwise(marks), strict=True)
    }


def rounded(usage: dict[str, float]) -> dict[str, float]:
    return {kind: round(milliseconds, 3) for kind, milliseconds in usage.items()}


def index_split() -> dict[str, dict[str, float]]:
    marks = [mark()]
    separate = faiss.read_index(str(artifact("search_index")), faiss.IO_FLAG_MMAP)
    marks.append(mark())
    separate.make_direct_map()
    marks.append(mark())
    return usages(["read_index", "make_direct_map"], marks)


def load_times() -> dict[str, dict[str, float]]:
    loads = (galaxy_count, labels, index, tokens, starts)
    marks = [mark()]
    for load in loads:
        load()
        marks.append(mark())
    return usages([load.__name__ for load in loads], marks)


def stage_times(query: Query, built: faiss.Index) -> dict[str, dict[str, float]]:
    marks = [mark()]
    direction = centroid(query, index=built)
    marks.append(mark())
    order = candidates(query, direction, index=built)
    marks.append(mark())
    rows = vectors(order, index=built)
    marks.append(mark())
    scored = score_maps(rows, direction, width=N_IMAGE_TOKENS)
    marks.append(mark())
    spectral_scores = span_maps(order, direction, index=built)
    marks.append(mark())
    scalar_scores = scalar_maps(order, direction, index=built)
    marks.append(mark())
    rank(order, scored, spectral_scores, scalar_scores)
    marks.append(mark())
    return usages(STAGES, marks)


def whole_searches(runs: int, matches: int, built: faiss.Index) -> dict[str, Any]:
    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    search(batch[0], index=built)
    return summary(
        [elapsed(partial(search, query, index=built)) for query in batch[1:]]
    )


@app.function(image=image)
def stages(runs: int, matches: int = 32) -> dict[str, Any]:
    split = index_split()
    loads = load_times()
    built = index()

    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    cold = stage_times(batch[0], built)
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in STAGES}

    for query in batch[1:]:
        for name, usage in stage_times(query, built).items():
            samples[name].append(usage)

    medians = {
        name: float(np.median([usage["wall"] for usage in values]))
        for name, values in samples.items()
    }
    means = {
        name: {
            kind: float(np.mean([usage[kind] for usage in values])) for kind in KINDS
        }
        for name, values in samples.items()
    }
    total = sum(medians.values())
    return {
        "environment": environment(),
        "load_ms": {name: round(usage["wall"], 3) for name, usage in loads.items()},
        "runs": runs,
        "matches": matches,
        "cold_stages_ms": {
            name: round(usage["wall"], 3) for name, usage in cold.items()
        },
        "total_p50_ms": round(total, 3),
        "stages": {
            name: {"p50_ms": round(median, 3), "share": round(median / total, 4)}
            for name, median in medians.items()
        },
        "searches": {
            f"matches={asked}": whole_searches(runs, asked, built) for asked in MATCHES
        },
        "usage_ms": {
            "index_split": {name: rounded(usage) for name, usage in split.items()},
            "loads": {name: rounded(usage) for name, usage in loads.items()},
            "stages": {name: rounded(mean) for name, mean in means.items()},
        },
        "memory": memory(),
    }


@app.function(
    image=image,
    cpu=SERVING_CPU,
    memory=SERVING_MEMORY,
    volumes={CACHE_PATH: cache_volume},
    timeout=6 * 60 * 60,
)
def benchmark_search_performance(
    sources: dict[str, bytes], order: list[str], runs: int
) -> dict[str, Any]:
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
    return report | {"rounds": rounds, "memory": memory()}


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
        "server": server(),
        "stages": benchmark_search_performance.remote(
            sources, names + names[::-1], runs
        )
        | {"locked_dependencies": "after"},
    }
    means = {
        name: float(np.mean([entry["total_p50_ms"] for entry in rounds]))
        for name, rounds in report["stages"]["rounds"].items()
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
