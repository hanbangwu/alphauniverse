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

    from app.alignment import AlignmentMap, apply_linear, held_out, pairs
    from app.config import PROVABGS, TEXT_SURVEYS, artifact, device
    from app.dataset import dataset
    from app.text_search import embed_queries

    saved = torch.load(artifact("alignment"), map_location=device())
    spaces = {}
    for modality in TEXT_SURVEYS:
        galaxy, aion, gemma = pairs(modality)
        validation = held_out(galaxy)
        aion, gemma = aion[validation], gemma[validation]
        model = AlignmentMap().to(device())
        model.load_state_dict(saved[modality]["mlp"])
        with torch.no_grad():
            documents = {
                "gemma": gemma,
                "linear": apply_linear(saved[modality]["linear"], aion),
                "mlp": model.eval()(aion),
            }
        held = galaxy[validation.cpu().numpy()]
        spaces |= {
            f"{modality} {name}": (held, rows.cpu().numpy())
            for name, rows in documents.items()
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
    scored = {
        (space, centring): (galaxy, transform(documents) @ transform(text).T)
        for space, (galaxy, documents) in spaces.items()
        for centring, transform in centrings.items()
    }
    for centring in centrings:
        galaxy, combined = scored[("image mlp", centring)]
        spectra, scores = scored[("spectrum mlp", centring)]
        combined = combined.copy()
        rows = np.searchsorted(galaxy, spectra)
        combined[rows] = np.maximum(combined[rows], scores)
        scored[("combined mlp", centring)] = (galaxy, combined)

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
        "galaxies": len(spaces["image mlp"][0]),
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
