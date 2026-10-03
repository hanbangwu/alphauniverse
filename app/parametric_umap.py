from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from umap import UMAP
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

from .config import (
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
    points,
)
from .dataset import dataset
from .search import source

BATCH = 1024
CHUNK = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
MIN_DIST = 0.1
NEGATIVES = 5
SAMPLE = 500_000


class ParametricUMAP(nn.Module):
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
        rows = torch.from_numpy(values)
        projected = [
            self(chunk.to(device())).cpu() for chunk in torch.split(rows, 4096)
        ]
        return torch.cat(projected).numpy()


def fit_parametric_umap(sampled: np.ndarray) -> ParametricUMAP:
    import wandb

    strengths, _, _ = fuzzy_simplicial_set(
        sampled, n_neighbors=UMAP().n_neighbors, random_state=SEED, metric="cosine"
    )
    edges = strengths.tocoo()
    rows = torch.as_tensor(sampled, device=device())
    heads = torch.as_tensor(edges.row, dtype=torch.int64, device=device())
    tails = torch.as_tensor(edges.col, dtype=torch.int64, device=device())
    strength = torch.as_tensor(edges.data, device=device())
    steps = len(strength) // BATCH

    a, b = find_ab_params(1.0, MIN_DIST)
    log_a = float(np.log(a))

    def logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return -(log_a + 2 * b * F.pairwise_distance(left, right).log())

    torch.manual_seed(SEED)
    model = ParametricUMAP(rows.shape[1]).to(device())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="train",
        tags=["parametric_umap"],
        group=DATASET_REVISION,
        mode=WANDB_MODE,
        config={
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH,
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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.detach()

            run.log({"epoch": epoch, "train/loss": total.item() / steps})

    torch.save(
        {"dim": rows.shape[1], "state": model.state_dict()}, artifact("parametric_umap")
    )
    return model.eval()


def _stream() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for survey in TOKEN_SURVEYS:
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


def embedding_count() -> int:
    metadata = pq.read_metadata(artifact("encoded"))
    columns = (
        metadata.row_group(group).column(leaf)
        for group in range(metadata.num_row_groups)
        for leaf in range(metadata.num_columns)
        if "." in metadata.schema.column(leaf).path
    )
    return (
        sum(column.num_values - column.statistics.null_count for column in columns)
        // DIM
    )


def generate_projections() -> None:
    category = (
        dataset().data.column(FLAG_SURVEYS["gz10"]).cast(pa.uint8()).combine_chunks()
    )
    count = len(category)
    galaxy = np.arange(count, dtype=np.int32)

    population = embedding_count()
    chosen = np.sort(
        np.random.default_rng(SEED).choice(
            population, min(SAMPLE, population), replace=False
        )
    )

    sums = np.zeros((count, DIM), dtype=np.float64)
    held = np.zeros(count, dtype=np.int64)
    taken: list[np.ndarray] = []
    seen = 0
    for galaxies, offsets, values in tqdm(_stream(), desc="scan"):
        sums[galaxies] += np.add.reduceat(values, offsets[:-1])
        held[galaxies] += np.diff(offsets)
        window = chosen[
            np.searchsorted(chosen, seen) : np.searchsorted(chosen, seen + len(values))
        ]
        taken.append(values[window - seen])
        seen += len(values)

    mean = (sums / held[:, None]).astype(dtype=np.float32)
    del sums, held

    sampled = np.concatenate(taken)
    taken.clear()
    model = fit_parametric_umap(sampled)
    del sampled

    pq.write_table(
        points(galaxy, model.transform(mean), category),
        artifact("mean_points"),
        compression="zstd",
    )

    with pq.ParquetWriter(
        artifact("full_points"), POINTS, compression="zstd"
    ) as writer:
        for galaxies, offsets, values in tqdm(_stream(), desc="project"):
            owner = np.repeat(galaxies, np.diff(offsets))
            writer.write_table(
                points(owner, model.transform(values), category.take(owner))
            )
