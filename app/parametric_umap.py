from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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
EPOCHS = 20
LEARNING_RATE = 1e-3
MIN_DIST = 0.1
NEGATIVES = 5
SAMPLE = 5_000_000
VALIDATION = 0.3
CURVE_A, CURVE_B = find_ab_params(1.0, MIN_DIST)


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


class Graph(NamedTuple):
    rows: torch.Tensor
    heads: torch.Tensor
    tails: torch.Tensor
    strength: np.ndarray


def split(sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(sampled, test_size=VALIDATION, random_state=SEED)


def graph(sampled: np.ndarray) -> Graph:
    strengths, _, _ = fuzzy_simplicial_set(
        sampled, n_neighbors=UMAP().n_neighbors, random_state=SEED, metric="cosine"
    )
    edges = strengths.tocoo()
    return Graph(
        torch.as_tensor(sampled, device=device()),
        torch.as_tensor(edges.row, dtype=torch.int64, device=device()),
        torch.as_tensor(edges.col, dtype=torch.int64, device=device()),
        edges.data / edges.data.sum(dtype=np.float64),
    )


def batches(edges: Graph, generator: np.random.Generator) -> tuple[torch.Tensor, ...]:
    steps = len(edges.strength) // BATCH
    draw = generator.choice(len(edges.strength), steps * BATCH, p=edges.strength)
    return torch.as_tensor(draw, device=device()).split(BATCH)


def loss(model: ParametricUMAP, edges: Graph, edge: torch.Tensor) -> torch.Tensor:
    def logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return -(
            float(np.log(CURVE_A))
            + 2 * CURVE_B * F.pairwise_distance(left, right).log()
        )

    anchor = model(edges.rows[edges.heads[edge]])
    positive = model(edges.rows[edges.tails[edge]])
    negative = model(
        edges.rows[
            torch.randint(len(edges.rows), (len(edge), NEGATIVES), device=device())
        ]
    )

    attract = logits(anchor, positive)
    repel = logits(anchor.unsqueeze(1), negative)
    return F.binary_cross_entropy_with_logits(
        attract, torch.ones_like(attract)
    ) + F.binary_cross_entropy_with_logits(repel, torch.zeros_like(repel))


def fit_parametric_umap(training: np.ndarray, validation: np.ndarray) -> ParametricUMAP:
    import wandb

    training_graph = graph(training)
    validation_graph = graph(validation)

    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    model = ParametricUMAP(training.shape[1]).to(device())
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
            "validation": VALIDATION,
            "seed": SEED,
        },
    ) as run:
        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch", summary="min")
        run.define_metric("validation/loss", step_metric="epoch", summary="min")

        for epoch in range(EPOCHS):
            training_losses = []
            for edge in tqdm(
                batches(training_graph, generator), desc="parametric umap"
            ):
                step = loss(model, training_graph, edge)
                optimizer.zero_grad()
                step.backward()
                optimizer.step()
                training_losses.append(step.detach())

            with torch.no_grad():
                validation_losses = [
                    loss(model, validation_graph, edge)
                    for edge in batches(validation_graph, generator)
                ]

            run.log(
                {
                    "epoch": epoch,
                    "train/loss": torch.stack(training_losses).mean().item(),
                    "validation/loss": torch.stack(validation_losses).mean().item(),
                }
            )

    torch.save(
        {"dim": training.shape[1], "state": model.state_dict()},
        artifact("parametric_umap"),
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
    table = source("encoded").to_table(columns=list(TOKEN_SURVEYS))
    return sum(
        pc.sum(pc.list_value_length(table.column(survey))).as_py()
        for survey in TOKEN_SURVEYS
    )


def sample() -> np.ndarray:
    population = embedding_count()
    return np.sort(
        np.random.default_rng(SEED).choice(
            population, min(SAMPLE, population), replace=False
        )
    )


def scan(count: int, chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(chosen)
    ascending = chosen[order]
    sums = np.zeros((count, DIM), dtype=np.float64)
    held = np.zeros(count, dtype=np.int64)
    taken = np.empty((len(chosen), DIM), dtype=np.float32)
    seen = 0
    for galaxies, offsets, values in tqdm(_stream(), desc="scan"):
        sums[galaxies] += np.add.reduceat(values, offsets[:-1])
        held[galaxies] += np.diff(offsets)
        first, last = np.searchsorted(ascending, (seen, seen + len(values)))
        taken[order[first:last]] = values[ascending[first:last] - seen]
        seen += len(values)
    if len(chosen) and ascending[-1] >= seen:
        raise ValueError(f"the sample reaches past the {seen} streamed embeddings")

    return (sums / held[:, None]).astype(dtype=np.float32), taken


def generate_projections() -> None:
    category = (
        dataset().data.column(FLAG_SURVEYS["gz10"]).cast(pa.uint8()).combine_chunks()
    )
    count = len(category)
    galaxy = np.arange(count, dtype=np.int32)

    training, validation = split(sample())
    mean, sampled = scan(count, np.concatenate((training, validation)))
    model = fit_parametric_umap(sampled[: len(training)], sampled[len(training) :])
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
