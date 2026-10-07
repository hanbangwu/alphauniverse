import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np

from app.config import DATASET_REVISION
from modal_app import CACHE_PATH, build_image, cache_volume, generate_alignment
from scripts.benchmarks.common import environment, git, spec

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
    gpu=generate_alignment.spec.gpus,
    cpu=generate_alignment.spec.cpu,
    memory=generate_alignment.spec.memory,
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality() -> dict[str, Any]:
    import torch
    from sklearn.metrics import average_precision_score

    from app.alignment import AlignmentMap, apply_linear, held_out, pairs
    from app.config import PROVABGS, artifact, device
    from app.dataset import dataset
    from app.text_search import embed_queries

    galaxy, aion, gemma = pairs()
    validation = held_out(galaxy)
    galaxy, aion, gemma = (
        galaxy[validation.cpu().numpy()],
        aion[validation],
        gemma[validation],
    )
    saved = torch.load(artifact("alignment"), map_location=device())
    model = AlignmentMap().to(device())
    model.load_state_dict(saved["mlp"])
    with torch.no_grad():
        spaces = {
            "gemma": gemma.cpu().numpy(),
            "linear": apply_linear(saved["linear"], aion).cpu().numpy(),
            "mlp": model.eval()(aion).cpu().numpy(),
        }

    columns = {"mass": "LOG_MSTAR", "sfr": "AVG_SFR", "redshift": "Z_HP"}
    data = dataset().select_columns(
        [f"{column}{PROVABGS}" for column in columns.values()]
    )[galaxy.tolist()]
    values = {
        name: np.asarray(data[f"{column}{PROVABGS}"], dtype=np.float64)
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
    corpora = {
        (space, centring): (transform(documents), transform(text))
        for space, documents in spaces.items()
        for centring, transform in centrings.items()
    }

    report = {}
    for index, (query, (column, low, high)) in enumerate(CUTS):
        population = ~np.isnan(values[column])
        relevant = ((values[column] > low) & (values[column] < high))[population]
        report[query] = {space: {} for space in spaces}
        for (space, centring), (documents, queries) in corpora.items():
            ranked = documents[population] @ queries[index]
            order = np.argsort(-ranked)
            report[query][space][centring] = {
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
        "galaxies": len(galaxy),
        "queries": report,
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "job": spec(benchmark_text_search_quality),
        **benchmark_text_search_quality.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
