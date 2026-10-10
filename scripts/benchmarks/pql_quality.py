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
    ANCHOR,
    DATASET_REVISION,
    DESI,
    REDSHIFT,
    SDSS,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
)
from app.dataset import dataset
from app.search import (
    FIRST_HSC_SCALAR,
    FIRST_LS_SCALAR,
    N_HSC_SCALARS,
    N_LS_SCALARS,
    Query,
    bounds,
    centroid,
    index,
    search,
    source,
    with_hsc,
    with_spectrum,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    environment,
    git,
    memory,
    observed_spans,
    wavelength,
)

app = modal.App("alphauniverse-pql-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/pql_quality.json")
WINDOW = 16
TABLE_VALUES = 4
TOP = 10
CANDIDATES = 128
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


def selections(galaxy: int, rng: np.random.Generator) -> dict[str, Query]:
    observed = observed_spans(wavelength(galaxy)).tolist()
    start = int(rng.integers(max(len(observed) - WINDOW, 0) + 1))
    kinds = {
        "spans_16": Query(galaxy=galaxy, spans=tuple(observed[start : start + WINDOW])),
        "spans_all": Query(galaxy=galaxy, spans=tuple(observed)),
        "ls_table": Query(
            galaxy=galaxy,
            scalars=tuple(
                (
                    FIRST_LS_SCALAR
                    + rng.choice(N_LS_SCALARS, TABLE_VALUES, replace=False)
                ).tolist()
            ),
        ),
    }
    if with_hsc()[galaxy]:
        chosen = FIRST_HSC_SCALAR + rng.choice(
            N_HSC_SCALARS, TABLE_VALUES, replace=False
        )
        kinds["hsc_table"] = Query(galaxy=galaxy, scalars=tuple(chosen.tolist()))
    return kinds


def hidden(galaxies: np.ndarray) -> tuple[pa.RecordBatch, list[np.ndarray]]:
    import torch

    from app import encode, predictions

    fitted = {survey: pql.basis()[survey][:2] for survey in SPECTRUM_SURVEYS}
    kept = torch.as_tensor(
        [
            encode.model().modality_info[key]["id"]
            for key in (
                predictions.IMAGES[ANCHOR],
                *predictions.SCALARS[ANCHOR],
                *predictions.SCALARS["hsc"],
            )
        ]
    )
    table = source("tokens").to_table(columns=["galaxy", *TOKEN_SURVEYS])
    records, rows = [], []
    for row in table.take(galaxies).to_pylist():
        row |= dict.fromkeys((*SPECTRUM_SURVEYS, REDSHIFT))
        encoded, _, mask, modality = encode.context(predictions.inputs(row))
        predicted = {
            key: predictions.predict(encoded, mask, key, positions)
            for key, positions in predictions.TARGETS.items()
        }
        records.append(predictions.record(row["galaxy"], predicted, fitted))
        tokens = encoded[0][torch.isin(modality[0], kept.to(modality.device))]
        rows.append(torch.nn.functional.normalize(tokens.float(), dim=-1).cpu().numpy())
    return pql.prediction_batch(records), rows


def best(rows: list[np.ndarray], direction: np.ndarray) -> np.ndarray:
    return np.asarray([(galaxy @ direction).max() for galaxy in rows])


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


def interval(pql_values: list[float], cosine_values: list[float]) -> dict[str, Any]:
    paired = np.asarray([pql_values, cosine_values], dtype=np.float64)
    paired = paired[:, np.isfinite(paired).all(axis=0)]
    differences = paired[0] - paired[1]
    if not len(differences):
        return {"queries": 0}
    draws = np.random.default_rng(0).integers(
        len(differences), size=(RESAMPLES, len(differences))
    )
    low, high = np.percentile(differences[draws].mean(axis=1), [2.5, 97.5])
    return {
        "queries": len(differences),
        "pql": round(float(paired[0].mean()), 4),
        "cosine": round(float(paired[1].mean()), 4),
        "difference": round(float(differences.mean()), 4),
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
    built = index()
    redshift = redshifts()
    known = np.isfinite(redshift)
    rng = np.random.default_rng(0)
    galaxies = np.sort(
        rng.choice(np.flatnonzero(with_spectrum() & known), sample, replace=False)
    )
    hidden_rows, hidden_tokens = hidden(galaxies)
    observed_tokens = [
        built.reconstruct_batch(np.arange(bounds()[galaxy], bounds()[galaxy + 1]))
        for galaxy in galaxies
    ]
    flips = rng.random((FLIPS, sample)) < 0.5

    measured: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"pql": [], "cosine": []})
    )
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
            found, cosine = search(
                query.model_copy(update={"matches": CANDIDATES}), index=built
            )[:2]
            matches = found[1:][known[found[1:]]]
            values["redshift"]["pql"].append(offset(redshift, order[:TOP], galaxy))
            values["redshift"]["cosine"].append(offset(redshift, matches[:TOP], galaxy))
            if not query.spans:
                continue

            selected = pql.selection(query)
            forms = pql.query_forms(pql.row(galaxy), selected)
            unseen = next(iter(pql.sums(hidden_rows, selected, forms).values()))
            direction = centroid(query, index=built)[0]
            unseen_cosine = best(hidden_tokens, direction)
            values["identity"]["pql"].append(
                float((np.delete(scores, galaxy) > unseen[position]).sum() < TOP)
            )
            values["identity"]["cosine"].append(
                float(unseen_cosine[position] > cosine[TOP])
            )
            for method, seen, missing in (
                ("pql", scores[galaxies], unseen),
                ("cosine", best(observed_tokens, direction), unseen_cosine),
            ):
                availability, evidence = bias(
                    seen[others], missing[others], near[others], flips[:, others]
                )
                values["availability"][method].append(availability)
                values["evidence"][method].append(evidence)

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
            kind: {
                metric: interval(pair["pql"], pair["cosine"])
                for metric, pair in metrics.items()
            }
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
