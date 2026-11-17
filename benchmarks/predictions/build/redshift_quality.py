import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
from astropy.stats import mad_std
from tqdm import tqdm

from app.config import (
    DATASET_REVISION,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    STORE_COLUMNS,
    artifact,
)
from app.encoding.store import tokens
from app.predictions.build import aion_predictions
from app.predictions.build import predictions as predictions_module
from app.predictions.build.observations import measured_redshifts, observed
from app.projection.build import properties
from benchmarks.common import environment, git, memory
from modal_app import CACHE_PATH, build_image, cache_volume

app = modal.App("alphauniverse-redshift-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/redshift_quality.json")
KEPT = ("ls", "hsc")
OUTLIER = 0.15
TARGET = {
    predictions_module.REDSHIFT_KEY: predictions_module.TARGETS[
        predictions_module.REDSHIFT_KEY
    ]
}


def hidden(galaxies: np.ndarray) -> np.ndarray:
    blank = dict.fromkeys(column for column in STORE_COLUMNS if column not in KEPT)
    found = []
    for row in tqdm(tokens().select(KEPT).take(galaxies).to_pylist(), desc="encode"):
        predicted = predictions_module.predictions(
            predictions_module.inputs(row | blank), TARGET
        )
        found.append(
            predictions_module.flushed(
                np.exp(predicted[predictions_module.REDSHIFT_KEY])
            )
        )
    return properties.spread_quantiles(
        np.stack(found)[:, 0], aion_predictions.bin_values(REDSHIFT)[0], 0.5
    )


def scored(medians: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    offsets = (medians - values) / (1 + values)
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
    galaxies = observed(REDSHIFT)
    values, surveys = measured_redshifts(galaxies)
    surveys = np.array(surveys)
    shipped = pq.read_table(artifact("properties"), columns=[f"{REDSHIFT}_predicted"])
    medians = {
        "stored": shipped.column(0).to_numpy()[galaxies],
        "hidden": hidden(galaxies),
    }
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "kept": KEPT,
        "outlier": OUTLIER,
        "inputs": {
            name: {"all": scored(found, values)}
            | {
                survey: scored(found[surveys == survey], values[surveys == survey])
                for survey in REDSHIFT_COLUMNS
            }
            for name, found in medians.items()
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
