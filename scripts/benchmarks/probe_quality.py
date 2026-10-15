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

from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    PROVABGS,
    REDSHIFT,
    SEED,
    SPECTRUM_MODES,
    artifact,
    galaxy_count,
)
from app.observations import column_values
from app.parametric_umap import scan
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-probe-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/probe_quality.json")
RANK = 256
NEIGHBOURS = 10
ALPHAS = np.logspace(-3, 4, 15)
FOLDS = KFold(shuffle=True, random_state=SEED)
SHOTS = (32, 128, 512)
DRAWS = 5
TARGETS = {"mass": "LOG_MSTAR", "sfr": "AVG_SFR", REDSHIFT: "Z_HP"}
BLOCKS = {
    "all": GRAM_NAMES,
    "without_spectra": tuple(
        name for name in GRAM_NAMES if not name.startswith(tuple(SPECTRUM_MODES))
    ),
}


def targets() -> dict[str, np.ndarray]:
    galaxies = np.arange(galaxy_count())
    values = {
        name: column_values(f"{column}{PROVABGS}", galaxies)
        for name, column in TARGETS.items()
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        values["sfr"] = np.log10(values["sfr"])
    return values


def scaled(role: str, name: str) -> np.ndarray:
    with np.load(artifact(f"{role}_{name}")) as stored:
        galleries = stored["galleries"][:, :RANK]
    centred = galleries - galleries.mean(axis=0)
    return centred / np.sqrt((centred**2).sum(axis=1).mean())


def shots(model: Any, features: np.ndarray, values: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    test, pool = np.split(rng.permutation(len(values)), [len(values) // 5])
    found = {}
    for count in SHOTS:
        fits = []
        for _ in range(DRAWS):
            train = rng.choice(pool, count, replace=False)
            fitted = model.fit(features[train], values[train])
            fits.append(r2_score(values[test], fitted.predict(features[test])))
        found[str(count)] = round(float(np.mean(fits)), 4)
    return found


def scores(features: np.ndarray, values: np.ndarray) -> dict[str, dict[str, Any]]:
    found = {}
    for label, model in (
        ("ridge", RidgeCV(alphas=ALPHAS)),
        ("knn", KNeighborsRegressor(NEIGHBOURS)),
    ):
        predicted = cross_val_predict(model, features, values, cv=FOLDS)
        found[label] = {
            "r2": round(float(r2_score(values, predicted)), 4),
            "rms": round(float(root_mean_squared_error(values, predicted)), 4),
            "shots": shots(model, features, values),
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
    aion = scan(galaxy_count(), np.empty(0, dtype=np.intp))[0]
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
                        [blocks[name][kept] for name in names if name != target],
                        axis=1,
                    ),
                    column[kept],
                )
                for label, names in BLOCKS.items()
            }
    return {
        "environment": environment(),
        "rank": RANK,
        "folds": FOLDS.n_splits,
        "neighbours": NEIGHBOURS,
        "draws": DRAWS,
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
