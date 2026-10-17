import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from umap import UMAP

from . import pql
from .config import FLAG_SURVEYS, MAPS, MODES, SEED, artifact, galaxy_count, points
from .dataset import dataset
from .embeddings import BATCH

NEIGHBOURS = UMAP().n_neighbors - 1


def nearest(anywhere: frozenset[str], count: int) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    for rows in tqdm(torch.arange(galaxy_count()).split(BATCH), desc="search"):
        totals = pql.whole_mode_totals(rows.numpy(), MODES, anywhere)
        scored = torch.from_numpy(sum(pql.contributions(totals).values()))
        scored[torch.arange(len(rows)), rows] = -torch.inf
        parts.append(scored.topk(count, dim=1))
    return (
        torch.cat([part.values for part in parts]),
        torch.cat([part.indices for part in parts]),
    )


def placed(
    found: tuple[torch.Tensor, torch.Tensor], features: np.ndarray
) -> np.ndarray:
    scores, indices = (part[:, :NEIGHBOURS].numpy() for part in found)
    own = np.arange(len(features))[:, None]
    knn = (
        np.hstack([own, indices]),
        np.hstack([np.zeros(own.shape), 1 + scores[:, :1] - scores]).astype(np.float32),
    )
    return UMAP(random_state=SEED, precomputed_knn=knn).fit_transform(features)


def generate_maps() -> None:
    features = np.load(artifact("joint_embedding"))
    category = (
        dataset().data.column(FLAG_SURVEYS["gz10"]).cast(pa.uint8()).combine_chunks()
    )
    galaxy = np.arange(len(category), dtype=np.int32)
    for name, anywhere in MAPS.items():
        pq.write_table(
            points(galaxy, placed(nearest(anywhere, NEIGHBOURS), features), category),
            artifact(f"{name}_points"),
            compression="zstd",
        )
