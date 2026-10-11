import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from app.config import DATASET_REVISION, N_IMAGE_TOKENS
from app.main import SPECTRUM_SURVEY
from app.pql import AGREEMENT_GALAXIES
from app.search import FIRST_LS_TABLE_VALUE, N_LS_TABLE_VALUES
from modal_app import SERVING_MAX_INPUTS, app, fastapi_app, serving_image
from scripts.benchmarks.common import (
    IMAGE_TOKENS,
    MATCHES,
    SPECTRUM_TOKENS,
    TABLE_VALUES,
    elapsed,
    environment,
    git,
    observed_spectrum_tokens,
    server,
    summary,
)
from scripts.benchmarks.text_search_quality import CUTS

image = serving_image.add_local_python_source("modal_app")

CLIENTS = (1, 4, SERVING_MAX_INPUTS, 2 * SERVING_MAX_INPUTS)
NEW_GALAXY_CLIENTS = 4
CACHED_PER_KIND = AGREEMENT_GALAXIES // 2
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

        def image_tokens(count: int) -> list[int]:
            return rng.choice(N_IMAGE_TOKENS, count, replace=False).tolist()

        def parameters(galaxy: int, matches: int) -> dict[str, Any]:
            return {
                "galaxy": galaxy,
                "ls_image": image_tokens(IMAGE_TOKENS),
                "matches": matches,
            }

        def similarity(galaxy: int, matches: int) -> None:
            get("/search", **parameters(galaxy, matches))

        cold_meta = round(elapsed(lambda: get("/meta")), 3)
        galaxies = session.get("/meta").raise_for_status().json()["galaxies"]
        unused = iter(rng.permutation(galaxies).tolist())

        def galaxy() -> int:
            return int(rng.integers(galaxies))

        cold_similarity = round(elapsed(lambda: similarity(next(unused), 32)), 3)
        cold_text_search = round(elapsed(lambda: get("/search/text", text=TEXTS[0])), 3)
        cells = pa.ipc.open_file(
            session.get("/downloads/tokens").raise_for_status().content
        ).read_all()
        has_spectrum = pc.is_valid(cells.column(SPECTRUM_SURVEY)).to_numpy()
        holders = np.flatnonzero(has_spectrum)
        random_galaxies = list(islice(unused, CACHED_PER_KIND))
        spectrum_galaxies = list(
            islice(
                (chosen for chosen in unused if has_spectrum[chosen]), CACHED_PER_KIND
            )
        )
        for chosen in random_galaxies + spectrum_galaxies:
            similarity(chosen, 32)
        random_rotation = cycle(random_galaxies)
        spectrum_rotation = cycle(spectrum_galaxies)
        texts = cycle(TEXTS)
        added = np.random.default_rng(1)

        calls = {
            "meta": lambda: get("/meta"),
            "image": lambda: get(f"/galaxy/{galaxy()}/image"),
            "tokens": lambda: get(f"/galaxy/{galaxy()}/image/tokens"),
            "coverage": lambda: get(f"/galaxy/{galaxy()}"),
            "table": lambda: get(f"/galaxy/{int(added.integers(galaxies))}/table"),
            "cosines": lambda: get(f"/galaxy/{galaxy()}/cosines/ls_image"),
            "text search": lambda: get("/search/text", text=next(texts)),
            "similarity table values matches=32": lambda: get(
                "/search",
                galaxy=next(random_rotation),
                table_values=(
                    FIRST_LS_TABLE_VALUE
                    + added.choice(N_LS_TABLE_VALUES, TABLE_VALUES, replace=False)
                ).tolist(),
            ),
            "saliency": lambda: get(
                "/saliency",
                galaxy=next(random_rotation),
                ls_image=image_tokens(IMAGE_TOKENS),
            ),
        } | {
            f"similarity matches={matches}": (
                lambda matches=matches: similarity(next(random_rotation), matches)
            )
            for matches in MATCHES
        }
        warm = {label: time_it(runs, call) for label, call in calls.items()}

        def concurrent(
            clients: int, count: int, timed_galaxies: Iterator[int]
        ) -> dict[str, Any]:
            batches = [
                [parameters(next(random_rotation), 32)]
                + [parameters(next(timed_galaxies), 32) for _ in range(count)]
                for _ in range(clients)
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

        def spectrum(route: str) -> None:
            get(f"/galaxy/{rng.choice(holders)}/spectrum{route}")

        def spectrum_token_query(image_token_count: int) -> dict[str, Any]:
            galaxy = next(spectrum_rotation)
            response = session.get(f"/galaxy/{galaxy}/spectrum")
            table = pa.ipc.open_stream(response.raise_for_status().content).read_all()
            spectrum_tokens = observed_spectrum_tokens(
                table.column("wavelength").to_numpy()
            )
            return {
                "galaxy": galaxy,
                "ls_image": image_tokens(image_token_count),
                f"{SPECTRUM_SURVEY}_spectrum": rng.choice(
                    spectrum_tokens, SPECTRUM_TOKENS, replace=False
                ).tolist(),
                "matches": 32,
            }

        def spectrum_token_similarity(image_token_count: int) -> Callable[[], None]:
            batch = iter(
                [spectrum_token_query(image_token_count) for _ in range(runs + 1)]
            )
            return lambda: get("/search", **next(batch))

        spectral_calls = {
            name: lambda route=route: spectrum(route)
            for name, route in (("spectrum", ""), ("spectrum tokens", "/tokens"))
        } | {
            "similarity spectrum tokens matches=32": spectrum_token_similarity(0),
            "similarity image tokens and spectrum tokens matches=32": spectrum_token_similarity(
                IMAGE_TOKENS
            ),
        }
        warm |= {label: time_it(runs, call) for label, call in spectral_calls.items()}
        concurrency = {
            f"clients={clients}": concurrent(clients, runs, random_rotation)
            for clients in CLIENTS
        } | {
            f"new galaxies clients={NEW_GALAXY_CLIENTS}": concurrent(
                NEW_GALAXY_CLIENTS, 1, unused
            )
        }
        new_galaxy = summary(
            timed_requests(
                session, [parameters(chosen, 32) for chosen in islice(unused, runs)]
            )
        )
        return {
            "environment": environment(),
            "cold_meta_ms": cold_meta,
            "cold_similarity_ms": cold_similarity,
            "cold_text_search_ms": cold_text_search,
            "warm": warm,
            "concurrency": concurrency,
            "new_galaxy": new_galaxy,
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
