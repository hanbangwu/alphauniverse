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

import numpy as np

from app import pql
from app.config import (
    DATASET_REVISION,
    N_SPECTRUM_TOKENS,
    OBSERVATIONS,
    galaxy_count,
    labels,
)
from app.search import FIRST_LS_TABLE_VALUE, N_LS_TABLE_VALUES
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
    TABLE_VALUES,
    elapsed,
    environment,
    git,
    memory,
    queries,
    server,
    spectrum_query,
    summary,
    with_spectrum,
)

image = serving_image.add_local_python_source("modal_app")

STAGES = [
    "forms",
    "scan",
    "combine",
    "similarity",
    "order",
    "maps",
    "predicted",
]
KINDS = ("wall", "user", "system")
SOURCES = ["app", "scripts", "modal_app.py"]
REPORT = Path("docs/benchmarks/search_performance.json")
SPECTRUM_TOKEN_WINDOW = 16
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


def load_times() -> dict[str, dict[str, float]]:
    loads = {
        "galaxy_count": galaxy_count,
        "labels": labels,
        "predictions": pql.predictions,
        "basis": pql.basis,
        **{
            f"observed_{column}": partial(pql.observed, column)
            for column in OBSERVATIONS
        },
    }
    marks = [mark()]
    for load in loads.values():
        load()
        marks.append(mark())
    return usages(list(loads), marks)


def stage_times(query: pql.Query) -> dict[str, dict[str, float]]:
    marks = [mark()]
    selected, forms = pql.selected_forms(query)
    marks.append(mark())
    totals = pql.scan(selected, forms)
    marks.append(mark())
    scored = pql.combine(totals)
    marks.append(mark())
    pql.similarity(pql.fractions(forms, totals))
    marks.append(mark())
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    marks.append(mark())
    pql.maps(query, galaxies)
    marks.append(mark())
    pql.predicted(query, galaxies)
    marks.append(mark())
    return usages(STAGES, marks)


def whole_searches(runs: int, matches: int) -> dict[str, Any]:
    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    pql.search(batch[0])
    return summary([elapsed(partial(pql.search, query)) for query in batch[1:]])


def kind_queries(count: int) -> dict[str, list[pql.Query]]:
    rng = np.random.default_rng(4)
    holders = rng.choice(np.flatnonzero(with_spectrum()), count)
    first = rng.integers(N_SPECTRUM_TOKENS - SPECTRUM_TOKEN_WINDOW + 1, size=count)
    return {
        "spectrum_tokens_16": [
            spectrum_query(
                int(galaxy), tuple(range(start, start + SPECTRUM_TOKEN_WINDOW))
            )
            for galaxy, start in zip(holders, first, strict=True)
        ],
        "spectrum_tokens_all": [
            spectrum_query(int(galaxy), tuple(range(N_SPECTRUM_TOKENS)))
            for galaxy in holders
        ],
        "table_values": [
            pql.Query(
                galaxy=int(rng.integers(galaxy_count())),
                table_values=tuple(
                    (
                        FIRST_LS_TABLE_VALUE
                        + rng.choice(N_LS_TABLE_VALUES, TABLE_VALUES, replace=False)
                    ).tolist()
                ),
            )
            for _ in range(count)
        ],
    }


def kind_times(runs: int, matches: int) -> dict[str, Any]:
    timed = {}
    for kind, batch in kind_queries(runs + 1).items():
        samples = []
        for query in batch:
            start = time.perf_counter()
            order = np.argsort(-pql.scores(query), kind="stable")[: matches + 1]
            scanned = time.perf_counter()
            pql.maps(query, order)
            samples.append(
                ((scanned - start) * 1000, (time.perf_counter() - scanned) * 1000)
            )
        timed[kind] = {
            "cold_ms": {
                "scan": round(samples[0][0], 3),
                "maps": round(samples[0][1], 3),
            },
            "scan": summary([scan for scan, _ in samples[1:]]),
            "maps": summary([maps for _, maps in samples[1:]]),
        }
    return timed


@app.function(image=image)
def stages(runs: int, matches: int = 32) -> dict[str, Any]:
    loads = load_times()

    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    cold = stage_times(batch[0])
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in STAGES}

    for query in batch[1:]:
        for name, usage in stage_times(query).items():
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
            f"matches={asked}": whole_searches(runs, asked) for asked in MATCHES
        },
        "kinds": kind_times(runs, matches),
        "usage_ms": {
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
