import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

from app.alignment import pairs
from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    PROVABGS,
    REDSHIFT,
    SPECTRUM_MODES,
    artifact,
)
from app.dataset import dataset
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-probe-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/probe_quality.json")
RANK = 256
FOLDS = 5
NEIGHBOURS = 10
SEED = 0
ALPHAS = np.logspace(-3, 4, 15)
TARGETS = {"mass": "LOG_MSTAR", "sfr": "AVG_SFR", "redshift": "Z_HP"}
BLOCKS = {
    "all": GRAM_NAMES,
    "without_spectra": tuple(
        name for name in GRAM_NAMES if not name.startswith(tuple(SPECTRUM_MODES))
    ),
}


def targets() -> dict[str, np.ndarray]:
    data = dataset().select_columns(
        [f"{column}{PROVABGS}" for column in TARGETS.values()]
    )
    values = {
        name: np.asarray(data[f"{column}{PROVABGS}"], dtype=np.float64)
        for name, column in TARGETS.items()
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        values["sfr"] = np.log10(values["sfr"])
    return values


def scaled(role: str, name: str) -> np.ndarray:
    galleries = np.load(artifact(f"{role}_{name}"))["galleries"][:, :RANK]
    centred = galleries - galleries.mean(axis=0)
    return centred / np.sqrt((centred**2).sum(axis=1).mean())


def scores(features: np.ndarray, values: np.ndarray) -> dict[str, dict[str, float]]:
    folds = KFold(FOLDS, shuffle=True, random_state=SEED)
    found = {}
    for label, model in (
        ("ridge", RidgeCV(alphas=ALPHAS)),
        ("knn", KNeighborsRegressor(NEIGHBOURS)),
    ):
        predicted = cross_val_predict(model, features, values, cv=folds)
        found[label] = {
            "r2": round(float(r2_score(values, predicted)), 4),
            "rms": round(float(root_mean_squared_error(values, predicted)), 4),
        }
    return found


@app.function(
    image=image,
    cpu=8,
    memory=(8 * 1024, 32 * 1024),
    timeout=30 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    values = targets()
    families = {
        role: {name: scaled(f"{role}_embedding", name) for name in GRAM_NAMES}
        for role in ("token", "template")
    }
    _, aion, _ = pairs()
    aion = aion.cpu().numpy()
    report = {}
    for target, column in values.items():
        kept = np.isfinite(column)
        report[target] = {
            "galaxies": int(kept.sum()),
            "aion": scores(aion[kept], column[kept]),
        }
        for role, blocks in families.items():
            report[target][role] = {
                label: scores(
                    np.concatenate(
                        [
                            blocks[name][kept]
                            for name in names
                            if not (target == "redshift" and name == REDSHIFT)
                        ],
                        axis=1,
                    ),
                    column[kept],
                )
                for label, names in BLOCKS.items()
            }
    return {
        "environment": environment(),
        "rank": RANK,
        "folds": FOLDS,
        "neighbours": NEIGHBOURS,
        "probes": report,
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
