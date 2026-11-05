import json
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow.compute as pc
from aion.modalities import LegacySurveyImage

from app.modal_app import app, fastapi_app, serving_image
from core.artifacts import source
from core.config import DATASET_REVISION, MODES, SPECTRUM_SURVEY
from core.modal_resources import CACHE_PATH, cache_volume
from scripts.benchmarks.common import (
    IMAGE_TOKENS,
    elapsed,
    environment,
    git,
    server,
    summary,
)

MATCHES = 32
REPORT = Path("docs/benchmarks/backend_performance.json")
TEXTS = (
    "a massive galaxy",
    "a dwarf galaxy",
    "a star-forming galaxy",
    "a quiescent galaxy with no ongoing star formation",
    "a distant galaxy at high redshift",
    "a nearby galaxy at low redshift",
)


def time_it(runs: int, call: Callable[[], Any]) -> dict[str, Any]:
    call()
    return summary([elapsed(call) for _ in range(runs)])


@app.function(
    image=serving_image, cpu=1, timeout=60 * 60, volumes={CACHE_PATH: cache_volume}
)
def benchmark_backend_performance(url: str, runs: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    texts = cycle(TEXTS)

    with httpx.Client(base_url=url, timeout=None) as session:

        def get(path: str, **params: Any) -> None:
            session.get(path, params=params).raise_for_status()

        cold = {"meta": round(elapsed(lambda: get("/meta")), 3)}
        galaxies = session.get("/meta").raise_for_status().json()["galaxy_count"]
        cells = source("tokens").to_table(columns=[SPECTRUM_SURVEY])
        holders = np.flatnonzero(pc.is_valid(cells.column(SPECTRUM_SURVEY)))

        def galaxy() -> int:
            return int(rng.integers(galaxies))

        calls = {
            "meta": lambda: get("/meta"),
            "search": lambda: get(
                "/search",
                galaxy=galaxy(),
                ls_image=rng.choice(
                    LegacySurveyImage.num_tokens, IMAGE_TOKENS, replace=False
                ).tolist(),
                matches=MATCHES,
            ),
            "whole-mode search": lambda: get(
                "/search/modes", galaxy=galaxy(), modes=list(MODES), matches=MATCHES
            ),
            "text search": lambda: get("/search/text", text=next(texts)),
            "image": lambda: get(f"/galaxy/{galaxy()}/image/ls"),
            "spectrum": lambda: get(
                f"/galaxy/{rng.choice(holders)}/spectrum/{SPECTRUM_SURVEY}"
            ),
            "table": lambda: get(f"/galaxy/{galaxy()}/table"),
        }
        cold |= {
            name: round(elapsed(call), 3)
            for name, call in calls.items()
            if name != "meta"
        }
        warm = {name: time_it(runs, call) for name, call in calls.items()}
        return {"environment": environment(), "cold_ms": cold, "warm": warm}


@app.local_entrypoint()
def main(runs: int = 30) -> None:
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
