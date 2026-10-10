import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from app import pql
from app.config import (
    DATASET_REVISION,
    DESI,
    REDSHIFT,
    SDSS,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
)
from app.dataset import dataset
from app.search import (
    FIRST_HSC_TABLE_VALUE,
    FIRST_LS_TABLE_VALUE,
    N_HSC_TABLE_VALUES,
    N_LS_TABLE_VALUES,
    source,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    environment,
    git,
    memory,
    observed_spectrum_tokens,
    spectrum_query,
    wavelength,
    with_spectrum,
)

app = modal.App("alphauniverse-pql-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/pql_quality.json")
WINDOW = 16
TABLE_VALUES = 4
TOP = 10
POOL = 32
FLIPS = 20
NEAR = 0.01
RESAMPLES = 10_000


def redshifts() -> np.ndarray:
    columns = dataset().data
    return (
        pc.coalesce(columns.column(f"Z{DESI}"), columns.column(f"Z{SDSS}"))
        .fill_null(np.nan)
        .to_numpy()
    )


def selections(galaxy: int, rng: np.random.Generator) -> dict[str, pql.Query]:
    observed = observed_spectrum_tokens(wavelength(galaxy)).tolist()
    start = int(rng.integers(max(len(observed) - WINDOW, 0) + 1))
    kinds = {
        "spectrum_tokens_16": spectrum_query(
            galaxy, tuple(observed[start : start + WINDOW])
        ),
        "spectrum_tokens_all": spectrum_query(galaxy, tuple(observed)),
        "ls_table": pql.Query(
            galaxy=galaxy,
            table_values=tuple(
                (
                    FIRST_LS_TABLE_VALUE
                    + rng.choice(N_LS_TABLE_VALUES, TABLE_VALUES, replace=False)
                ).tolist()
            ),
        ),
    }
    if pql.observed("hsc")[galaxy]:
        chosen = FIRST_HSC_TABLE_VALUE + rng.choice(
            N_HSC_TABLE_VALUES, TABLE_VALUES, replace=False
        )
        kinds["hsc_table"] = pql.Query(
            galaxy=galaxy, table_values=tuple(chosen.tolist())
        )
    return kinds


def hidden(galaxies: np.ndarray) -> pa.RecordBatch:
    from app import encode, predictions

    fitted = {survey: pql.basis()[survey][:2] for survey in SPECTRUM_SURVEYS}
    table = source("tokens").to_table(columns=["galaxy", *TOKEN_SURVEYS])
    records = []
    for row in table.take(galaxies).to_pylist():
        row |= dict.fromkeys((*SPECTRUM_SURVEYS, REDSHIFT))
        encoded, _, mask, _ = encode.context(predictions.inputs(row))
        predicted = predictions.decode(encoded, mask, predictions.TARGETS)
        records.append(predictions.record(row["galaxy"], predicted, fitted))
    return pql.prediction_batch(records)


def bias(
    seen: np.ndarray, unseen: np.ndarray, near: np.ndarray, flips: np.ndarray
) -> tuple[float, float]:
    availability, evidence = [], []
    for flip in flips:
        top = np.argsort(-np.where(flip, seen, unseen), kind="stable")[:POOL]
        for pool, shares in ((~near, availability), (near, evidence)):
            picked = top[pool[top]]
            if len(picked) and pool.any():
                shares.append(flip[picked].mean() - flip[pool].mean())
    return (
        float(np.mean(availability)) if availability else np.nan,
        float(np.mean(evidence)) if evidence else np.nan,
    )


def offset(redshift: np.ndarray, chosen: np.ndarray, galaxy: int) -> float:
    return float(
        np.median(np.abs(redshift[chosen] - redshift[galaxy]) / (1 + redshift[galaxy]))
    )


def interval(values: list[float]) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"queries": 0}
    draws = np.random.default_rng(0).integers(
        len(finite), size=(RESAMPLES, len(finite))
    )
    low, high = np.percentile(finite[draws].mean(axis=1), [2.5, 97.5])
    return {
        "queries": len(finite),
        "mean": round(float(finite.mean()), 4),
        "low": round(float(low), 4),
        "high": round(float(high), 4),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=6 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_pql_quality(sample: int) -> dict[str, Any]:
    redshift = redshifts()
    known = np.isfinite(redshift)
    rng = np.random.default_rng(0)
    galaxies = np.sort(
        rng.choice(np.flatnonzero(with_spectrum() & known), sample, replace=False)
    )
    hidden_rows = hidden(galaxies)
    flips = rng.random((FLIPS, sample)) < 0.5

    measured: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for position, galaxy in enumerate(galaxies.tolist()):
        others = np.arange(sample) != position
        near = (
            np.abs(redshift[galaxies] - redshift[galaxy]) / (1 + redshift[galaxy])
            < NEAR
        )
        for kind, query in selections(galaxy, rng).items():
            values = measured[kind]
            scores = pql.scores(query)
            order = np.argsort(-scores, kind="stable")
            order = order[(order != galaxy) & known[order]]
            values["redshift"].append(offset(redshift, order[:TOP], galaxy))
            if not (query.desi_spectrum or query.sdss_spectrum):
                continue

            selected, forms = pql.selected_forms(query)
            unseen = next(iter(pql.sums(hidden_rows, selected, forms).values()))
            values["identity"].append(
                float((np.delete(scores, galaxy) > unseen[position]).sum() < TOP)
            )
            availability, evidence = bias(
                scores[galaxies][others],
                unseen[others],
                near[others],
                flips[:, others],
            )
            values["availability"].append(availability)
            values["evidence"].append(evidence)

    return {
        "environment": environment(),
        "sample": sample,
        "window": WINDOW,
        "table_values": TABLE_VALUES,
        "top": TOP,
        "pool": POOL,
        "flips": FLIPS,
        "near": NEAR,
        "results": {
            kind: {metric: interval(values) for metric, values in metrics.items()}
            for kind, metrics in measured.items()
        },
        "memory": memory(),
    }


@app.local_entrypoint()
def main(sample: int = 1000) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_pql_quality.remote(sample),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
