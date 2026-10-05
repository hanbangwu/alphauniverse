import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np

from app.config import DATASET_REVISION, SEED
from modal_app import CACHE_PATH, build_image, cache_volume, generate_projections
from scripts.benchmarks.common import environment, git, spec

app = modal.App("alphauniverse-projection")
image = build_image.add_local_python_source("modal_app")

NEIGHBOURS = (15, 100)
SIZE = 10_000
REPORT = Path("docs/benchmarks/projection_quality.json")


def preservation(embedded: np.ndarray, projected: np.ndarray, count: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    high = (
        NearestNeighbors(n_neighbors=count, metric="cosine")
        .fit(embedded)
        .kneighbors(return_distance=False)
    )
    low = (
        NearestNeighbors(n_neighbors=count)
        .fit(projected)
        .kneighbors(return_distance=False)
    )
    shared = [
        len(np.intersect1d(near, far)) for near, far in zip(high, low, strict=True)
    ]
    return float(np.mean(shared)) / count


@app.function(
    image=image,
    cpu=generate_projections.spec.cpu,
    memory=generate_projections.spec.memory,
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_projection_quality() -> dict[str, Any]:
    import torch
    from sklearn.manifold import trustworthiness

    from app.config import artifact, device, galaxy_count
    from app.parametric_umap import ParametricUMAP, scan, split

    _, sampled = scan(galaxy_count())
    _, validation = split(sampled)
    embedded = np.random.default_rng(SEED).choice(validation, SIZE, replace=False)

    saved = torch.load(artifact("parametric_umap"), map_location=device())
    model = ParametricUMAP(saved["dim"]).to(device())
    model.load_state_dict(saved["state"])
    projected = model.eval().transform(embedded)

    return {
        "environment": environment(),
        "size": SIZE,
        "neighbours": {
            count: {
                "preservation": round(preservation(embedded, projected, count), 4),
                "trustworthiness": round(
                    float(
                        trustworthiness(
                            embedded, projected, n_neighbors=count, metric="cosine"
                        )
                    ),
                    4,
                ),
            }
            for count in NEIGHBOURS
        },
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "job": spec(benchmark_projection_quality),
        **benchmark_projection_quality.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
