import io
import json
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
from app.config import DATASET_REVISION
from modal_app import (
    CACHE_PATH,
    SERVING_CPU,
    SERVING_ENVIRONMENT,
    SERVING_MEMORY,
    app,
    cache_volume,
    serving_image,
)
from scripts.benchmarks.common import (
    IMAGE_TOKENS,
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
    "selection",
    "forms",
    "sums",
    "combine",
    "similarity",
    "order",
    "maps",
    "predicted",
]
MATCHES = (32, 64, 128)
SAMPLES = 4
SOURCES = ["app", "scripts", "modal_app.py"]
REPORT = Path("docs/benchmarks/search_performance.json")
ENTRY = (
    "import json, sys\n"
    "from app.config import DATASET_REVISION\n"
    "try:\n"
    "    from scripts.benchmarks.search_performance import stages\n"
    "except ModuleNotFoundError:\n"
    "    from scripts.benchmark import stages\n"
    "print(json.dumps(stages.local(int(sys.argv[1]))"
    " | {'revision': DATASET_REVISION}))"
)


def stage_times(query: pql.Query) -> dict[str, float]:
    marks = [time.perf_counter()]
    selected = pql.selection(query)
    marks.append(time.perf_counter())
    forms = pql.galaxy_forms(query.galaxy)
    peaks = pql.slot_peaks(forms)
    marks.append(time.perf_counter())
    totals = pql.mode_totals(query.galaxy, selected, {})
    marks.append(time.perf_counter())
    scored = pql.combine(totals)
    marks.append(time.perf_counter())
    pql.similarity(pql.fractions(pql.selected_peaks(peaks, selected), totals))
    marks.append(time.perf_counter())
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    marks.append(time.perf_counter())
    pql.maps(forms, peaks, pql.gathered(galaxies))
    marks.append(time.perf_counter())
    pql.predicted(selected, galaxies)
    marks.append(time.perf_counter())
    return {
        name: (stop - start) * 1000
        for name, (start, stop) in zip(STAGES, pairwise(marks), strict=True)
    }


def matrix_times(batch: list[pql.Query]) -> dict[str, Any]:
    pql.MATRICES.clear()
    builds = [elapsed(partial(pql.agreements, query.galaxy)) for query in batch]
    clicks = [elapsed(partial(pql.saliency, query)) for query in batch]
    return {"agreements": summary(builds), "saliency": summary(clicks)}


def whole_searches(runs: int, matches: int) -> dict[str, Any]:
    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    pql.search(batch[0])
    return summary([elapsed(partial(pql.search, query)) for query in batch[1:]])


def anywhere_times(runs: int) -> dict[str, Any]:
    searches, clicks, repeats = [], [], []
    for query in queries(runs + 1, IMAGE_TOKENS, MATCHES[0]):
        anywhere = query.model_copy(update={"anywhere": ("ls_image",)})
        searches.append(elapsed(partial(pql.search, anywhere)))
        clicks.append(elapsed(partial(pql.saliency, anywhere)))
        repeats.append(elapsed(partial(pql.saliency, anywhere)))
    return {
        "search": summary(searches[1:]),
        "saliency": summary(clicks[1:]),
        "repeat_saliency": summary(repeats[1:]),
    }


@app.function(image=image)
def stages(runs: int, matches: int = MATCHES[0]) -> dict[str, Any]:
    batch = queries(runs + 1, IMAGE_TOKENS, matches)
    stage_times(batch[0])
    samples = [stage_times(query) for query in batch[1:]]
    medians = {
        name: float(np.median([sample[name] for sample in samples])) for name in STAGES
    }
    total = sum(medians.values())
    clicked = batch[1 : SAMPLES + 1]
    return {
        "environment": environment(),
        "runs": runs,
        "matches": matches,
        "total_p50_ms": round(total, 3),
        "stages": {
            name: {
                "p50_ms": round(median, 3),
                "share": round(median / total, 4),
            }
            for name, median in medians.items()
        },
        "matrices": matrix_times(clicked),
        "searches": {
            f"matches={asked}": whole_searches(runs, asked) for asked in MATCHES
        },
        "anywhere": anywhere_times(runs),
        "memory": memory(),
    }


@app.function(
    image=image,
    env=SERVING_ENVIRONMENT,
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
    for name, source_archive in sources.items():
        with tarfile.open(fileobj=io.BytesIO(source_archive)) as bundle:
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
    commits = {"current": git("rev-parse", "HEAD")}
    current_code = git("rev-parse", *(f"HEAD:{source}" for source in SOURCES))
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
            if stored_code != current_code:
                commits["best"] = stored["commit"]
    names = list(commits)
    sources = {
        name: subprocess.check_output(["git", "archive", commit, *SOURCES])
        for name, commit in commits.items()
    }
    timed = benchmark_search_performance.remote(
        sources, names + names[::-1] if len(names) > 1 else names, runs
    )
    report = {
        "commit": git("describe", "--always"),
        "commits": commits,
        "notes": notes,
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": server(),
        "stages": timed | {"locked_dependencies": "current"},
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
