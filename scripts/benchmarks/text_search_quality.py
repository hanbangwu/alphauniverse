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


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality() -> dict[str, Any]:
    import torch
    from sklearn.metrics import average_precision_score

    from app.alignment import apply_linear, held_out, pairs, trained
    from app.config import PROVABGS, TEXT_SURVEYS, artifact, device
    from app.dataset import dataset
    from app.text_search import embed_queries

    saved = torch.load(artifact("alignment"), map_location=device())
    held, spaces = {}, {}
    for modality in TEXT_SURVEYS:
        galaxy, aion, gemma = pairs(modality)
        validation = held_out(galaxy)
        aion, gemma = aion[validation], gemma[validation]
        held[modality] = galaxy[validation.cpu().numpy()]
        with torch.no_grad():
            documents = {
                "gemma": gemma,
                "linear": apply_linear(saved[modality]["linear"], aion),
                "mlp": trained(saved[modality]["mlp"])(aion),
            }
        spaces |= {
            (modality, name): rows.cpu().numpy() for name, rows in documents.items()
        }

    columns = {"mass": "LOG_MSTAR", "sfr": "AVG_SFR", "redshift": "Z_HP"}
    values = {
        name: np.asarray(dataset().data.column(f"{column}{PROVABGS}"), dtype=np.float64)
        for name, column in columns.items()
    }
    with np.errstate(divide="ignore"):
        values["specific_sfr"] = np.log10(values.pop("sfr")) - values["mass"]

    text = embed_queries([query for query, _ in CUTS])
    centrings = {
        "raw": lambda rows: rows,
        "centred": lambda rows: (
            (rows - rows.mean(axis=0))
            / np.linalg.norm(rows - rows.mean(axis=0), axis=-1, keepdims=True)
        ),
    }
    scored = {}
    for centring, transform in centrings.items():
        queries = transform(text)
        combined = np.full((len(dataset()), len(CUTS)), -np.inf)
        for (modality, name), documents in spaces.items():
            scores = transform(documents) @ queries.T
            scored[(f"{modality} {name}", centring)] = (held[modality], scores)
            if name == "mlp":
                combined[held[modality]] = np.maximum(combined[held[modality]], scores)
        everyone = np.unique(np.concatenate(list(held.values())))
        scored[("combined mlp", centring)] = (everyone, combined[everyone])

    report = {}
    for index, (query, (column, low, high)) in enumerate(CUTS):
        report[query] = {}
        for (space, centring), (galaxy, scores) in scored.items():
            value = values[column][galaxy]
            population = ~np.isnan(value)
            relevant = ((value > low) & (value < high))[population]
            ranked = scores[population, index]
            order = np.argsort(-ranked)
            report[query].setdefault(space, {})[centring] = {
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
