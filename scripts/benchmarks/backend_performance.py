import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from app.config import DATASET_REVISION, N_PATCHES
from app.main import SPECTRUM_SURVEY
from app.search import FIRST_LS_SCALAR, N_LS_SCALARS
from modal_app import SERVING_MAX_INPUTS, app, fastapi_app, serving_image
from scripts.benchmarks.common import (
    MATCHES,
    PATCHES,
    SCALARS,
    SPANS,
    elapsed,
    environment,
    git,
    observed_spans,
    server,
    summary,
)
from scripts.benchmarks.text_search_quality import CUTS

image = serving_image.add_local_python_source("modal_app")

CLIENTS = (1, 4, SERVING_MAX_INPUTS, 2 * SERVING_MAX_INPUTS)
REPORT = Path("docs/benchmarks/backend_performance.json")
TEXTS = tuple(text for text, _ in CUTS)


def time_it(runs: int, call: Callable[[], Any]) -> dict[str, Any]:
    call()
    return summary([elapsed(call) for _ in range(runs)])


def timed_requests(session: httpx.Client, batch: list[dict[str, Any]]) -> list[float]:
    def ask(params: dict[str, Any]) -> None:
        session.get("/search", params=params).raise_for_status()

    return [elapsed(partial(ask, params)) for params in batch]


@app.function(image=image, cpu=1, timeout=60 * 60)
def benchmark_backend_performance(url: str, runs: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)

    with httpx.Client(base_url=url, timeout=None) as session:

        def get(path: str, **params: Any) -> None:
            session.get(path, params=params).raise_for_status()

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
        cold_text_search = round(elapsed(lambda: get("/search/text", text=TEXTS[0])), 3)
        texts = cycle(TEXTS)
        added = np.random.default_rng(1)

        calls = {
            "meta": lambda: get("/meta"),
            "image": lambda: get(f"/galaxy/{galaxy()}/image"),
            "tokens": lambda: get(f"/galaxy/{galaxy()}/image/tokens"),
            "coverage": lambda: get(f"/galaxy/{galaxy()}"),
            "table": lambda: get(f"/galaxy/{int(added.integers(galaxies))}/table"),
            "text search": lambda: get("/search/text", text=next(texts)),
            "similarity scalars matches=32": lambda: get(
                "/search",
                galaxy=int(added.integers(galaxies)),
                t=(
                    FIRST_LS_SCALAR + added.choice(N_LS_SCALARS, SCALARS, replace=False)
                ).tolist(),
            ),
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

        cells = pa.ipc.open_file(
            session.get("/downloads/tokens").raise_for_status().content
        ).read_all()
        holders = np.flatnonzero(pc.is_valid(cells.column(SPECTRUM_SURVEY)).to_numpy())

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
            "cold_text_search_ms": cold_text_search,
            "warm": warm
            | {label: time_it(runs, call) for label, call in spectral_calls.items()},
            "concurrency": {
                f"clients={clients}": concurrent(clients) for clients in CLIENTS
            },
        }


@app.local_entrypoint()
def main(runs: int = 30) -> None:
    if git("status", "--porcelain"):
        raise SystemExit("Commit every change first: the benchmark times commits.")
    report = {
        "commit": git("describe", "--always"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": server(),
        "requests": benchmark_backend_performance.remote(
            fastapi_app.get_web_url(), runs
        ),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
