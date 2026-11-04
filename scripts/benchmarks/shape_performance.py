import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import numpy as np

from app import pql
from app.config import DATASET_REVISION, GRID, N_IMAGE_TOKENS, SPECTRUM_SURVEYS
from app.search import FIRST_LS_TABLE_VALUE
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
    environment,
    git,
    memory,
    observed_spectrum_tokens,
    server,
    summary,
    wavelength,
    with_spectrum,
)

image = serving_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/shape_performance.json")
SIZES = (2, 3, 8, 30)
MIXED = 3
TABLE_VALUES = tuple(range(FIRST_LS_TABLE_VALUE, FIRST_LS_TABLE_VALUE + 4))
MATCHES = 32
CELLS = np.stack(np.divmod(np.arange(N_IMAGE_TOKENS), GRID), axis=-1)


def compact_cells(rng: np.random.Generator, count: int) -> np.ndarray:
    centre = rng.uniform(GRID / 4, 3 * GRID / 4, 2)
    distances = np.linalg.norm(CELLS - centre, axis=1)
    return np.sort(np.argsort(distances, kind="stable")[:count])


def spread_cells(rng: np.random.Generator, count: int) -> np.ndarray:
    return np.sort(rng.choice(N_IMAGE_TOKENS, count, replace=False))


def compact_windows(
    rng: np.random.Generator, windows: np.ndarray, count: int
) -> np.ndarray:
    start = rng.integers(len(windows) - count + 1)
    return windows[start : start + count]


def spread_windows(
    rng: np.random.Generator, windows: np.ndarray, count: int
) -> np.ndarray:
    return np.sort(rng.choice(windows, count, replace=False))


def spectrum_mode(galaxy: int) -> str:
    survey = next(survey for survey in SPECTRUM_SURVEYS if pql.observed(survey)[galaxy])
    return f"{survey}_spectrum"


def overlap(found: np.ndarray, expected: np.ndarray) -> float:
    return len(np.intersect1d(found[1:], expected[1:])) / (len(expected) - 1)


def poses(mode: str, slots: tuple[int, ...], shift: pql.Shift) -> int:
    reads = pql.pose_reads(mode, slots, shift)
    return len(reads.indices) // (
        1 if isinstance(reads, pql.ImageReads) else len(slots)
    )


def measured(
    galaxy: int,
    selected: dict[str, np.ndarray],
    shapes: list[str],
    shift: pql.Shift,
) -> dict[str, Any]:
    keys = {mode: tuple(selected[mode].tolist()) for mode in shapes}
    for cached in (pql.basket_sum, pql.shape_sum, pql.pose_reads):
        cached.cache_clear()
    start = time.perf_counter()
    for mode, slots in keys.items():
        pql.basket_sum(galaxy, mode, slots)
    bound = time.perf_counter()
    exact = {
        mode: pql.shape_sum(galaxy, mode, slots, shift).totals
        for mode, slots in keys.items()
    }
    done = time.perf_counter()
    aligned = {mode: slots for mode, slots in selected.items() if mode not in shapes}
    ranking = pql.ranked(
        pql.combine(pql.column_sums(galaxy, aligned) | exact), galaxy, MATCHES
    )
    for cached in (pql.basket_sum, pql.pose_reads):
        cached.cache_clear()
    began = time.perf_counter()
    found = pql.pruned(galaxy, selected, shapes, shift, MATCHES)
    finished = time.perf_counter()
    return {
        "poses": sum(poses(mode, slots, shift) for mode, slots in keys.items()),
        "bound_ms": (bound - start) * 1000,
        "exact_ms": (done - bound) * 1000,
        "pruned_ms": (finished - began) * 1000,
        "evaluated": found.evaluated,
        "agreement": overlap(found.galaxies, ranking),
        "ranking": ranking,
    }


Selected = tuple[int, dict[str, np.ndarray], list[str]]


def selections(queries: int) -> dict[str, list[Selected]]:
    rng = np.random.default_rng(0)
    imaged = np.arange(pql.predictions().num_rows)
    spectra = np.flatnonzero(with_spectrum())
    found: dict[str, list[Selected]] = {}
    for count in SIZES:
        for layout, cells in (("compact", compact_cells), ("spread", spread_cells)):
            found[f"ls_image {layout} {count}"] = [
                (int(galaxy), {"ls_image": cells(rng, count)}, ["ls_image"])
                for galaxy in rng.choice(imaged, queries, replace=False)
            ]
    for count in SIZES:
        for layout, windows in (
            ("compact", compact_windows),
            ("spread", spread_windows),
        ):
            batch = []
            for galaxy in rng.choice(spectra, queries, replace=False):
                mode = spectrum_mode(int(galaxy))
                observed = observed_spectrum_tokens(wavelength(int(galaxy)))
                batch.append(
                    (int(galaxy), {mode: windows(rng, observed, count)}, [mode])
                )
            found[f"spectrum {layout} {count}"] = batch
    both, tables = [], []
    for galaxy in rng.choice(spectra, queries, replace=False):
        mode = spectrum_mode(int(galaxy))
        observed = observed_spectrum_tokens(wavelength(int(galaxy)))
        cells = compact_cells(rng, MIXED)
        both.append(
            (
                int(galaxy),
                {"ls_image": cells, mode: compact_windows(rng, observed, MIXED)},
                ["ls_image", mode],
            )
        )
        query = pql.Query(
            galaxy=int(galaxy),
            ls_image=tuple(cells.tolist()),
            table_values=TABLE_VALUES,
        )
        tables.append((int(galaxy), pql.selection(query), ["ls_image"]))
    found[f"ls_image and spectrum compact {MIXED}"] = both
    found[f"ls_image compact {MIXED} and table values"] = tables
    return found


def described(runs: list[dict[str, Any]], galaxies: int) -> dict[str, Any]:
    return {
        "poses": int(np.median([run["poses"] for run in runs])),
        "bound": summary([run["bound_ms"] for run in runs]),
        "exact": summary([run["exact_ms"] for run in runs]),
        "pruned": summary([run["pruned_ms"] for run in runs]),
        "evaluated_share": round(
            float(np.median([run["evaluated"] for run in runs])) / galaxies, 4
        ),
        "pruned_agreement": round(
            float(np.mean([run["agreement"] for run in runs])), 4
        ),
    }


@app.function(
    image=image,
    env=SERVING_ENVIRONMENT,
    cpu=SERVING_CPU,
    memory=SERVING_MEMORY,
    volumes={CACHE_PATH: cache_volume},
    timeout=3 * 60 * 60,
)
def measure(queries: int) -> dict[str, Any]:
    galaxies = pql.predictions().num_rows
    configs = {}
    for name, batch in selections(queries).items():
        shifts = get_args(pql.Shift) if "ls_image" in name else ("cell",)
        runs = {shift: [measured(*query, shift) for query in batch] for shift in shifts}
        configs[name] = {
            shift: described(found, galaxies) for shift, found in runs.items()
        }
        if "half" in runs:
            configs[name]["against_half"] = {
                shift: round(
                    float(
                        np.mean(
                            [
                                overlap(run["ranking"], half["ranking"])
                                for run, half in zip(
                                    runs[shift], runs["half"], strict=True
                                )
                            ]
                        )
                    ),
                    4,
                )
                for shift in ("cell", "refined")
            }
    return {
        "environment": environment(),
        "galaxies": galaxies,
        "queries": queries,
        "matches": MATCHES,
        "configs": configs,
        "memory": memory(),
    }


@app.local_entrypoint()
def main(queries: int = 5) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": server(),
        **measure.remote(queries),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
