import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np

from app.config import DATASET_REVISION
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-text-search")
image = build_image.add_local_python_source("modal_app")

CUTS = (
    ("A massive galaxy, stellar mass above 10^11 solar masses", ("mass", 11, np.inf)),
    ("A low-mass dwarf galaxy", ("mass", -np.inf, 9.5)),
    ("A star-forming galaxy", ("specific_sfr", -10, np.inf)),
    (
        "A quiescent galaxy with no ongoing star formation",
        ("specific_sfr", -np.inf, -11),
    ),
    ("A distant galaxy at redshift above 0.5", ("redshift", 0.5, np.inf)),
    ("A nearby galaxy at redshift below 0.1", ("redshift", -np.inf, 0.1)),
)
TOP = (10, 100)
REPORT = Path("docs/benchmarks/text_search_quality.json")


def centred(rows: np.ndarray) -> np.ndarray:
    shifted = rows - rows.mean(axis=0)
    return shifted / np.linalg.norm(shifted, axis=-1, keepdims=True)


@app.function(
    image=image,
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality() -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    from app.config import PROVABGS
    from app.dataset import dataset
    from app.text_search import DOCUMENTS, embed_queries, gemma_space

    space = gemma_space()
    everyone = np.unique(space.galaxy)
    columns = {"mass": "LOG_MSTAR", "sfr": "AVG_SFR", "redshift": "Z_HP"}
    values = {
        name: np.asarray(dataset().data.column(f"{column}{PROVABGS}"), dtype=np.float64)
        for name, column in columns.items()
    }
    with np.errstate(divide="ignore"):
        values["specific_sfr"] = np.log10(values.pop("sfr")) - values["mass"]

    text = embed_queries([query for query, _ in CUTS])
    centrings = {"raw": lambda rows: rows, "centred": centred}
    scored = {}
    for centring, transform in centrings.items():
        queries = transform(text)
        combined = np.full((len(dataset()), len(CUTS)), -np.inf)
        for modality in DOCUMENTS:
            rows = space.modality == modality
            galaxy = space.galaxy[rows]
            scores = transform(space.vectors[rows]) @ queries.T
            scored[(modality, centring)] = (galaxy, scores)
            combined[galaxy] = np.maximum(combined[galaxy], scores)
        scored[("combined", centring)] = (everyone, combined[everyone])

    report = {}
    for index, (query, (column, low, high)) in enumerate(CUTS):
        report[query] = {}
        for (name, centring), (galaxy, scores) in scored.items():
            value = values[column][galaxy]
            population = ~np.isnan(value)
            relevant = ((value > low) & (value < high))[population]
            ranked = scores[population, index]
            order = np.argsort(-ranked)
            report[query].setdefault(name, {})[centring] = {
                "base_rate": round(float(relevant.mean()), 4),
                "average_precision": round(
                    float(average_precision_score(relevant, ranked)), 4
                ),
            } | {
                f"precision@{top}": round(float(relevant[order[:top]].mean()), 4)
                for top in TOP
            }

    return {
        "environment": environment(),
        "galaxies": len(everyone),
        "queries": report,
        "memory": memory(),
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_text_search_quality.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
