import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
from astropy.stats import mad_std
from tqdm import tqdm

from app import pql, predictions, templates
from app.config import DATASET_REVISION, REDSHIFT, REDSHIFT_COLUMNS, STORE_COLUMNS
from app.observations import measured_redshifts
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-redshift-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/redshift_quality.json")
KEPT = ("ls", "hsc")
OUTLIER = 0.15
TARGET = {predictions.REDSHIFT_KEY: predictions.TARGETS[predictions.REDSHIFT_KEY]}


def stored(galaxies: np.ndarray) -> np.ndarray:
    table = pql.predictions().select([REDSHIFT]).take(galaxies)
    return np.concatenate(
        [pql.table_value_probabilities(rows, REDSHIFT) for rows in table.to_batches()]
    )


def hidden(galaxies: np.ndarray) -> np.ndarray:
    blank = dict.fromkeys(column for column in STORE_COLUMNS if column not in KEPT)
    found = []
    for row in tqdm(tokens().take(galaxies).to_pylist(), desc="encode"):
        predicted = predictions.predictions(predictions.inputs(row | blank), TARGET)
        found.append(predictions.flushed(np.exp(predicted[predictions.REDSHIFT_KEY])))
    return np.stack(found)


def scored(probabilities: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    bins = templates.bin_values(REDSHIFT)
    median = templates.quantiles(probabilities, bins, 0.5)[:, 0]
    offsets = (median - values) / (1 + values)
    return {
        "galaxies": len(values),
        "bias": round(float(np.median(offsets)), 4),
        "nmad": round(float(mad_std(offsets)), 4),
        "outlier_rate": round(float(np.mean(np.abs(offsets) > OUTLIER)), 4),
    }


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=(16 * 1024, 32 * 1024),
    timeout=20 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    galaxies = np.flatnonzero(pql.observed(REDSHIFT))
    values, surveys = measured_redshifts(galaxies)
    surveys = np.array(surveys)
    inputs = {"stored": stored(galaxies), "hidden": hidden(galaxies)}
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "kept": KEPT,
        "outlier": OUTLIER,
        "inputs": {
            name: {"all": scored(probabilities, values)}
            | {
                survey: scored(
                    probabilities[surveys == survey], values[surveys == survey]
                )
                for survey in REDSHIFT_COLUMNS
            }
            for name, probabilities in inputs.items()
        },
        "memory": memory(),
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **measure.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
