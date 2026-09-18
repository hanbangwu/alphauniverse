"""Stage 3 of the build: project embeddings to 2-d with a parametric UMAP.

A small MLP is trained to reproduce UMAP's fuzzy-simplicial-set structure on a
sample of embeddings, then applied to everything. Because the projection is a
function rather than a fitted table, the same model maps both point sets:
`mean_points` (one point per galaxy, from its mean embedding) and
`full_points` (one point per embedding, all modalities in the same space).
"""

from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

from .config import (
    DATASET_ID,
    DATASET_REVISION,
    DIM,
    FLAG_SURVEYS,
    POINTS,
    SEED,
    TOKEN_SURVEYS,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    artifact,
    device,
)
from .dataset import dataset
from .search import source

BATCH = 1024
CHUNK = 128
EPOCHS = 10
LR = 1e-3
NEIGHBORS = 16
MIN_DIST = 0.1
NEGATIVES = 5
SAMPLE = 500_000

class ParametricUMAP(nn.Module):
    """An MLP mapping a normalised embedding to 2-d."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.encoder(F.normalize(rows, dim=-1))

    @torch.inference_mode()
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Project rows to 2-d in chunks, returning coordinates on the CPU."""
        fitted = next(self.parameters()).device
        rows = torch.from_numpy(values)
        projected = [self(chunk.to(fitted)).cpu() for chunk in torch.split(rows, 4096)]
        return torch.cat(projected).numpy()


def fit_parametric_umap(sampled: np.ndarray) -> None:
    """Train the projector on sampled embeddings and save its weights.

    Edges come from UMAP's fuzzy simplicial set over `sampled`; each step
    samples edges by strength and pulls their endpoints together while pushing
    `NEGATIVES` random rows apart.
    """
    import wandb

    strengths, _, _ = fuzzy_simplicial_set(
        sampled, n_neighbors=NEIGHBORS, random_state=SEED, metric="cosine"
    )
    edges = strengths.tocoo()
    rows = torch.as_tensor(sampled, device=device())
    heads = torch.as_tensor(edges.row, dtype=torch.int64, device=device())
    tails = torch.as_tensor(edges.col, dtype=torch.int64, device=device())
    strength = torch.as_tensor(edges.data, dtype=torch.float32, device=device())
    steps = len(strength) // BATCH

    a, b = find_ab_params(1.0, MIN_DIST)
    log_a = float(np.log(a))

    def logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return -(log_a + 2 * b * F.pairwise_distance(left, right).log())

    torch.manual_seed(SEED)
    model = ParametricUMAP(rows.shape[1]).to(device())
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="train",
        tags=["parametric_umap"],
        group=DATASET_REVISION,
        mode=WANDB_MODE,
        config={
            "optimizer": "Adam",
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "n_neighbors": NEIGHBORS,
            "min_dist": MIN_DIST,
            "negatives": NEGATIVES,
            "sample": SAMPLE,
            "seed": SEED,
        },
    ) as run:
        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch", summary="min")

        for epoch in range(EPOCHS):
            draw = torch.multinomial(strength, steps * BATCH, replacement=True)
            total = torch.zeros((), device=device())
            for edge in tqdm(draw.split(BATCH), desc="parametric umap"):
                anchor = model(rows[heads[edge]])
                positive = model(rows[tails[edge]])
                negative = model(
                    rows[
                        torch.randint(
                            len(rows), (len(edge), NEGATIVES), device=device()
                        )
                    ]
                )

                attract = logits(anchor, positive)
                repel = logits(anchor.unsqueeze(1), negative)
                loss = F.binary_cross_entropy_with_logits(
                    attract, torch.ones_like(attract)
                ) + F.binary_cross_entropy_with_logits(repel, torch.zeros_like(repel))

                optim.zero_grad()
                loss.backward()
                optim.step()

                total += loss.detach()

            run.log({"epoch": epoch, "train/loss": total.item() / steps})

    model.eval()
    weights = artifact("parametric_umap")
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dim": rows.shape[1], "state": model.state_dict()}, weights)


def _stream(
    counts: dict[str, int],
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream `encoded` survey by survey, skipping galaxies with no match.

    Yields `(galaxies, offsets, values)` per batch, where `offsets` bound
    each galaxy's slice of `values`.
    """
    for survey in TOKEN_SURVEYS:
        if not counts[survey]:
            continue
        scanner = source("encoded").scanner(
            columns=["galaxy", survey],
            batch_size=CHUNK,
            filter=ds.field(survey).is_valid(),
        )
        for batch in scanner.to_batches():
            cells = batch.column(survey)
            offsets = np.asarray(cells.offsets, dtype=np.intp)
            yield (
                np.asarray(batch.column("galaxy")),
                offsets - offsets[0],
                np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
                    -1, DIM
                ),
            )


def _points(galaxy: np.ndarray, coords: np.ndarray, category: pa.Array) -> pa.Table:
    """A table of projected points in the `POINTS` schema."""
    return pa.table(
        {
            "galaxy": pa.array(galaxy),
            "x": pa.array(coords[:, 0]),
            "y": pa.array(coords[:, 1]),
            "category": category,
        },
        schema=POINTS,
    )


def generate_projections() -> None:
    """Train the projector and write both point sets.

    Streams `encoded` once to accumulate per-galaxy embedding means and draw a
    `SAMPLE`-sized training set, then streams it again to project every
    embedding into `full_points`.
    """
    for role in ("parametric_umap", "mean_points", "full_points"):
        path = artifact(role)
        path.unlink(missing_ok=True)
        path.with_name(f"{path.name}.partial").unlink(missing_ok=True)

    data = dataset(DATASET_ID, DATASET_REVISION)
    raw = np.asarray(data[FLAG_SURVEYS["gz10"]], dtype=np.float64)
    category = pa.array(np.nan_to_num(raw).astype(dtype=np.uint8), mask=np.isnan(raw))
    count = len(category)
    galaxy = np.arange(count, dtype=np.int32)

    metadata = pq.read_metadata(artifact("encoded"))
    counts: dict[str, int] = {}
    for leaf in range(len(metadata.schema)):
        survey, _, nested = metadata.schema.column(leaf).path.partition(".")
        if not nested:
            continue
        chunks = [
            metadata.row_group(group).column(leaf)
            for group in range(metadata.num_row_groups)
        ]
        counts[survey] = (
            sum(chunk.num_values for chunk in chunks)
            - sum(chunk.statistics.null_count for chunk in chunks)
        ) // DIM

    population = sum(counts.values())
    chosen = np.sort(
        np.random.default_rng(SEED).choice(
            population, min(SAMPLE, population), replace=False
        )
    )

    sums = np.zeros((count, DIM), dtype=np.float64)
    held = np.zeros(count, dtype=np.int64)
    taken: list[np.ndarray] = []
    seen = 0
    for gids, offsets, values in tqdm(_stream(counts), desc="scan"):
        live = np.diff(offsets) > 0
        sums[gids[live]] += np.add.reduceat(values, offsets[:-1][live])
        held[gids] += np.diff(offsets)
        window = chosen[
            np.searchsorted(chosen, seen) : np.searchsorted(chosen, seen + len(values))
        ]
        taken.append(values[window - seen])
        seen += len(values)

    mean = (sums / held[:, None]).astype(dtype=np.float32)
    del sums, held

    sampled = np.concatenate(taken)
    taken.clear()
    fit_parametric_umap(sampled)
    del sampled

    saved = torch.load(artifact("parametric_umap"), map_location="cpu")
    model = ParametricUMAP(saved["dim"])
    model.load_state_dict(saved["state"])
    model.to(device()).eval()

    pq.write_table(
        _points(galaxy, model.transform(mean), category),
        artifact("mean_points"),
        compression="zstd",
    )

    full_points = artifact("full_points")
    staging = full_points.with_name(f"{full_points.name}.partial")
    with pq.ParquetWriter(staging, POINTS, compression="zstd") as writer:
        for gids, offsets, values in tqdm(_stream(counts), desc="project"):
            owner = np.repeat(gids, np.diff(offsets))
            writer.write_table(
                _points(owner, model.transform(values), category.take(owner))
            )

    staging.replace(full_points)
