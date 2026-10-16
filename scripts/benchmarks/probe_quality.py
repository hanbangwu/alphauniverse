import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from app import embeddings
from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    PROVABGS,
    REDSHIFT,
    SEARCH_RANK,
    SEED,
    SPECTRUM_MODES,
    artifact,
    galaxy_count,
)
from app.observations import column_values
from app.parametric_umap import scan
from app.properties import ALPHAS, FOLDS, provabgs
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-probe-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/probe_quality.json")
NEIGHBOURS = 10
PROBES = {"ridge": RidgeCV(alphas=ALPHAS), "knn": KNeighborsRegressor(NEIGHBOURS)}
SHOTS = (32, 128, 512)
DRAWS = 20
BLOCKS = {
    "all": GRAM_NAMES,
    "without_spectra": tuple(
        name for name in GRAM_NAMES if not name.startswith(tuple(SPECTRUM_MODES))
    ),
}


def targets() -> dict[str, np.ndarray]:
    galaxies = np.arange(galaxy_count())
    return provabgs(galaxies) | {REDSHIFT: column_values(f"Z_HP{PROVABGS}", galaxies)}


def galleries(role: str, name: str) -> np.ndarray:
    with np.load(artifact(f"{role}_{name}")) as stored:
        return stored["galleries"][:, :SEARCH_RANK]


def shots(
    model: Any, features: np.ndarray, values: np.ndarray
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    test, pool = np.split(rng.permutation(len(values)), [len(values) // 5])
    found = {}
    for count in SHOTS:
        fits, alphas = [], []
        for _ in range(DRAWS):
            train = rng.choice(pool, count, replace=False)
            fitted = model.fit(features[train], values[train])
            fits.append(r2_score(values[test], fitted.predict(features[test])))
            if isinstance(fitted, RidgeCV):
                alphas.append(fitted.alpha_)
        found[str(count)] = {
            "r2": round(float(np.mean(fits)), 4),
            "sd": round(float(np.std(fits, ddof=1)), 4),
        }
        if alphas:
            found[str(count)] |= {
                "alpha": float(np.quantile(alphas, 0.5, method="lower")),
                "alpha_at_edge": round(
                    float(np.isin(alphas, ALPHAS[[0, -1]]).mean()), 4
                ),
            }
    return found


def stacked(widths: dict[str, int]) -> StackingRegressor:
    edges = np.cumsum([0, *widths.values()])
    return StackingRegressor(
        [
            (
                name,
                make_pipeline(
                    ColumnTransformer([(name, "passthrough", np.arange(start, end))]),
                    RidgeCV(alphas=ALPHAS),
                ),
            )
            for name, start, end in zip(widths, edges[:-1], edges[1:], strict=True)
        ],
        final_estimator=LinearRegression(positive=True),
        cv=FOLDS,
    )


def scores(
    features: np.ndarray, values: np.ndarray, models: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    found = {}
    for label, model in models.items():
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
        role: {
            name: embeddings.scaled(galleries(f"{role}_embedding", name))
            for name in GRAM_NAMES
        }
        for role in ("token", "template")
    }
    aion = scan(galaxy_count(), np.empty(0, dtype=np.intp))[0]
    report = {}
    for target, column in values.items():
        kept = np.isfinite(column)
        report[target] = {
            "galaxies": int(kept.sum()),
            "aion": scores(aion[kept], column[kept], PROBES),
            "modes": {},
        }
        for role, blocks in families.items():
            features = {
                label: np.concatenate(
                    [blocks[name] for name in names if name != target], axis=1
                )
                for label, names in BLOCKS.items()
            }
            features["joint"] = embeddings.joint(
                [blocks[name] for name in GRAM_NAMES if name != target]
            )
            report[target][role] = {
                label: scores(matrix[kept], column[kept], PROBES)
                for label, matrix in features.items()
            }
            widths = {
                name: block.shape[1] for name, block in blocks.items() if name != target
            }
            report[target][role]["stacked"] = scores(
                features["all"][kept], column[kept], {"ridge": stacked(widths)}
            )
            fitted = stacked(widths).fit(features["all"][kept], column[kept])
            report[target]["modes"][role] = {
                name: {
                    "weight": round(float(weight), 4),
                    "shots": shots(PROBES["ridge"], blocks[name][kept], column[kept]),
                }
                for name, weight in zip(
                    widths, fitted.final_estimator_.coef_, strict=True
                )
            }
    return {
        "environment": environment(),
        "rank": SEARCH_RANK,
        "joint": embeddings.JOINT_RANK,
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
