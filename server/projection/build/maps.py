import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from umap import UMAP

from alphauniverse.config import MAPS, MODES, SEED, artifact, points
from alphauniverse.encoding.store import object_ids
from alphauniverse.predictions.build.embeddings import BATCH
from alphauniverse.predictions.whole_mode import contributions, whole_mode_totals

NEIGHBORS = UMAP().n_neighbors - 1


def nearest(
    galaxies: int, anywhere: frozenset[str], count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    for rows in tqdm(torch.arange(galaxies).split(BATCH), desc="search"):
        totals = whole_mode_totals(rows.numpy(), MODES, anywhere)
        scored = torch.from_numpy(sum(contributions(totals).values()))
        scored[torch.arange(len(rows)), rows] = -torch.inf
        parts.append(scored.topk(count, dim=1))
    return (
        torch.cat([part.values for part in parts]),
        torch.cat([part.indices for part in parts]),
    )


def placed(
    found: tuple[torch.Tensor, torch.Tensor], features: np.ndarray
) -> np.ndarray:
    scores, indices = (part[:, :NEIGHBORS].numpy() for part in found)
    own = np.arange(len(features))[:, None]
    knn = (
        np.hstack([own, indices]),
        np.hstack([np.zeros(own.shape), 1 + scores[:, :1] - scores]).astype(np.float32),
    )
    return UMAP(random_state=SEED, precomputed_knn=knn).fit_transform(features)


def generate_maps() -> None:
    features = np.load(artifact("joint_embedding"))
    for name, anywhere in MAPS.items():
        pq.write_table(
            points(
                object_ids(),
                placed(nearest(len(features), anywhere, NEIGHBORS), features),
            ),
            artifact(f"{name}_points"),
            compression="zstd",
        )
