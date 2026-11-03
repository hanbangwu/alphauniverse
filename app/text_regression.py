import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.optim.swa_utils import AveragedModel

from . import embeddings, text_views
from .config import (
    DATASET_REVISION,
    GEMMA_DIM,
    GRAM_NAMES,
    MODES,
    PARTS_PER_ROUND,
    PROTOTYPE_HELD_OUT,
    SEED,
    TEXT_CONDITION,
    TEXT_RANK,
    TOKEN_GRAMS,
    VIEW_COMBINATIONS,
    VIEW_TYPES,
    VIEWS_PER_PART,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    artifact,
    device,
)
from .text_search import embed_queries, generic_queries

HIDDEN = 512
KIND_WIDTH = 16
BATCH = 512
MASKED = 0.3
DROPPED = 0.1
CHUNK = 400
STEPS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DECAY = 0.999
PATIENCE = 10
IMPROVEMENT = 0.001
SHARED_BASIS = 128
CAP = 1_000_000
GALAXIES = 1024

Bases = dict[str, torch.Tensor]


class HeldOut(NamedTuple):
    galaxies: torch.Tensor
    rows: torch.Tensor
    owners: torch.Tensor
    view_types: torch.Tensor
    directions: torch.Tensor


class Regression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kind = nn.Embedding(len(VIEW_TYPES), KIND_WIDTH)
        self.layers = nn.Sequential(
            nn.Linear(TEXT_CONDITION + len(GRAM_NAMES) + KIND_WIDTH, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, TEXT_RANK),
        )

    def forward(
        self, condition: torch.Tensor, present: torch.Tensor, kind: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(torch.cat([condition, present, self.kind(kind)], dim=-1))


def views(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = pq.read_table(path, columns=["galaxy", "view_type", "embedding"])
    embedding = table.column("embedding").combine_chunks().flatten().to_numpy()
    return (
        torch.tensor(table.column("galaxy").to_numpy(), device=device()).long(),
        torch.tensor(table.column("view_type").to_numpy(), device=device()).long(),
        torch.tensor(embedding.reshape(len(table), GEMMA_DIM), device=device()).float(),
    )


def first_round() -> tuple[torch.Tensor, torch.Tensor]:
    count = PARTS_PER_ROUND * VIEWS_PER_PART
    view_types = torch.empty(count, dtype=torch.long, device=device())
    embeddings = torch.empty(count, GEMMA_DIM, device=device())
    for index in range(PARTS_PER_ROUND):
        rows = slice(index * VIEWS_PER_PART, (index + 1) * VIEWS_PER_PART)
        _, view_types[rows], embeddings[rows] = views(text_views.part_path(index))
    return view_types, embeddings


def residuals(
    mean: torch.Tensor, view_types: torch.Tensor, embeddings: torch.Tensor
) -> torch.Tensor:
    return F.normalize(embeddings - mean[view_types], dim=-1)


def fitted_bases(view_types: torch.Tensor, embeddings: torch.Tensor) -> Bases:
    mean = torch.stack(
        [embeddings[view_types == kind].mean(dim=0) for kind in range(len(VIEW_TYPES))]
    )
    found = residuals(mean, view_types, embeddings).cpu().numpy()
    kinds = view_types.cpu().numpy()
    components, centre, scale = [], [], []
    for kind, (_, name) in enumerate(VIEW_TYPES):
        fitted = kinds == kind
        if fitted.sum() < SHARED_BASIS:
            siblings = [
                other for other, view in enumerate(VIEW_TYPES) if view[1] == name
            ]
            fitted = np.isin(kinds, siblings)
        pca = PCA(TEXT_RANK, random_state=SEED).fit(found[fitted])
        components.append(pca.components_)
        centre.append(pca.mean_)
        scale.append(pca.transform(found[fitted]).std(axis=0))
    return {
        "mean": mean,
        "components": torch.from_numpy(np.stack(components)).to(device()),
        "centre": torch.from_numpy(np.stack(centre)).to(device()),
        "scale": torch.from_numpy(np.stack(scale)).to(device()),
    }


def whitening(rows: torch.Tensor) -> Bases:
    pca = PCA(TEXT_CONDITION, whiten=True, random_state=SEED).fit(rows.cpu().numpy())
    projection = pca.components_.T / np.sqrt(pca.explained_variance_)
    return {
        "offset": torch.from_numpy(pca.mean_).to(device()),
        "projection": torch.from_numpy(projection).to(device()),
    }


def coefficients(
    bases: Bases, view_types: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    found = torch.empty(len(directions), TEXT_RANK, device=directions.device)
    for kind in view_types.unique():
        chosen = view_types == kind
        centred = directions[chosen] - bases["centre"][kind]
        found[chosen] = centred @ bases["components"][kind].T / bases["scale"][kind]
    return found


def condition(rows: torch.Tensor, present: torch.Tensor, bases: Bases) -> torch.Tensor:
    masked = rows.unflatten(1, (len(GRAM_NAMES), -1)) * present[..., None]
    return (masked.flatten(1) - bases["offset"]) @ bases["projection"]


def training_weights(view_types: torch.Tensor) -> torch.Tensor:
    draws = torch.tensor(
        [
            text_views.SAMPLE_WEIGHTS[name] if kind == "sample" else 1
            for kind, name in VIEW_TYPES
        ],
        device=view_types.device,
    )
    return 1 / draws[view_types]


def masks(count: int) -> torch.Tensor:
    present = (torch.rand(count, len(GRAM_NAMES), device=device()) > MASKED).float()
    return present * (torch.rand(count, 1, device=device()) > DROPPED)


def decayed(
    averages: list[torch.Tensor], currents: list[torch.Tensor], count: torch.Tensor
) -> None:
    weight = 1 - torch.clamp((1 + count) / (10 + count), max=DECAY)
    for average, current in zip(averages, currents, strict=True):
        average.lerp_(current, weight)


def stepped(
    network: Regression,
    average: AveragedModel,
    optimiser: torch.optim.Optimizer,
    weights: torch.Tensor,
    inputs: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
) -> float:
    losses = []
    for _ in range(STEPS):
        target, *conditions = inputs(
            torch.multinomial(weights, BATCH, replacement=True)
        )
        loss = ((network(*conditions) - target) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        average.update_parameters(network)
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


@torch.no_grad()
def predicted(
    network: Regression, bases: Bases, rows: torch.Tensor, present: torch.Tensor
) -> torch.Tensor:
    conditions = condition(rows, present, bases)
    found = torch.empty(
        len(VIEW_TYPES), len(rows), bases["components"].shape[2], device=rows.device
    )
    for kind in range(len(VIEW_TYPES)):
        kinds = torch.full((len(rows),), kind, device=rows.device)
        points = network(conditions, present, kinds) * bases["scale"][kind]
        found[kind] = points @ bases["components"][kind] + bases["centre"][kind]
    return found


def identification(
    network: Regression, held: HeldOut, bases: Bases, present: torch.Tensor
) -> torch.Tensor:
    predictions = predicted(
        network, bases, held.rows, present.expand(len(held.galaxies), -1)
    )
    own = torch.searchsorted(held.galaxies, held.owners)
    beaten = torch.empty(len(held.owners), device=held.rows.device)
    for kind in held.view_types.unique():
        chosen = held.view_types == kind
        scores = held.directions[chosen] @ predictions[kind].T
        mine = scores.gather(1, own[chosen, None])
        below = (scores < mine).sum(dim=1) + ((scores == mine).sum(dim=1) - 1) / 2
        beaten[chosen] = below / (len(held.galaxies) - 1)
    sums = beaten.new_zeros(len(VIEW_TYPES)).index_add_(0, held.view_types, beaten)
    return sums / torch.bincount(held.view_types, minlength=len(VIEW_TYPES))


def evaluated(joint: torch.Tensor, bases: Bases) -> HeldOut:
    owners, view_types, found = views(artifact("text_views_held_out"))
    galaxies = owners.unique()
    return HeldOut(
        galaxies,
        joint[galaxies],
        owners,
        view_types,
        residuals(bases["mean"], view_types, found),
    )


def subset(held: HeldOut, galaxies: torch.Tensor) -> HeldOut:
    kept = torch.isin(held.galaxies, galaxies)
    chosen = torch.isin(held.owners, galaxies)
    return HeldOut(
        held.galaxies[kept],
        held.rows[kept],
        held.owners[chosen],
        held.view_types[chosen],
        held.directions[chosen],
    )


def patterns() -> dict[str, torch.Tensor]:
    found = {}
    for mode in MODES:
        blocks = torch.tensor(
            [name in TOKEN_GRAMS[mode] for name in GRAM_NAMES],
            dtype=torch.float32,
            device=device(),
        )
        found[f"without {mode}"] = 1 - blocks
        found[f"{mode} alone"] = blocks
    return found


def per_type(found: torch.Tensor) -> dict[str, float]:
    values = found.cpu().numpy()
    return {"mean": round(float(np.nanmean(values)), 4)} | {
        f"{kind} {name}": round(float(value), 4)
        for (kind, name), value in zip(VIEW_TYPES, values, strict=True)
        if np.isfinite(value)
    }


def report(network: Regression, held: HeldOut, bases: Bases) -> dict:
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    prototype = subset(held, torch.from_numpy(PROTOTYPE_HELD_OUT).to(device()))
    return {
        "identification": per_type(identification(network, held, bases, everything)),
        "prototype identification": per_type(
            identification(network, prototype, bases, everything)
        ),
        "masked": {
            name: per_type(identification(network, held, bases, present[None]))
            for name, present in patterns().items()
        },
    }


def train_text_regression() -> None:
    import wandb

    text_views.generate_text_views(PARTS_PER_ROUND)
    joint = embeddings.loaded("joint_embedding", torch.float32)
    bases = fitted_bases(*first_round()) | whitening(joint[text_views.training()])
    held = evaluated(joint, bases)
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    targets = torch.empty(CAP, TEXT_RANK, device=device())
    owners = torch.empty(CAP, dtype=torch.long, device=device())
    kinds = torch.empty(CAP, dtype=torch.long, device=device())
    torch.manual_seed(SEED)
    network = Regression().to(device())
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    average = AveragedModel(network, multi_avg_fn=decayed)

    def inputs(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        present = masks(BATCH)
        rows = joint[owners[batch]]
        return targets[batch], condition(rows, present, bases), present, kinds[batch]

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="train",
        tags=["text_regression"],
        group=DATASET_REVISION,
        mode=WANDB_MODE,
        config={
            "rank": TEXT_RANK,
            "condition": TEXT_CONDITION,
            "hidden": HIDDEN,
            "batch": BATCH,
            "chunk": CHUNK,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "masked": MASKED,
            "dropped": DROPPED,
            "seed": SEED,
        },
    ) as run:
        filled, best, best_part = 0, -1.0, 0
        for index in range(CAP // VIEWS_PER_PART):
            if not text_views.part_path(index).exists():
                rounds = index // PARTS_PER_ROUND + 1
                text_views.generate_text_views(rounds * PARTS_PER_ROUND)
            galaxy, view_type, embedding = views(text_views.part_path(index))
            rows = slice(filled, filled + len(galaxy))
            directions = residuals(bases["mean"], view_type, embedding)
            targets[rows] = coefficients(bases, view_type, directions)
            owners[rows], kinds[rows] = galaxy, view_type
            losses = []
            for start in range(filled, rows.stop, CHUNK):
                weights = training_weights(kinds[: min(start + CHUNK, rows.stop)])
                losses.append(stepped(network, average, optimiser, weights, inputs))
            filled = rows.stop
            overall = identification(average.module, held, bases, everything).nanmean()
            entry = {
                "views": filled,
                "train/loss": float(np.mean(losses)),
                "identification": overall.item(),
            }
            run.log(entry)
            print(json.dumps(entry), flush=True)
            if overall.item() > best + IMPROVEMENT:
                best, best_part, best_views = overall.item(), index, filled
                best_network = deepcopy(average.module)
            if index - best_part >= PATIENCE:
                break
        found = {"views": best_views} | report(best_network, held, bases)
        run.summary.update(found)
        print(json.dumps(found, indent=1))
    torch.save(best_network.state_dict(), artifact("text_regression"))
    np.savez(
        artifact("text_bases"),
        **{name: value.cpu().numpy() for name, value in bases.items()},
    )


def stored_regression() -> tuple[Regression, Bases]:
    with np.load(artifact("text_bases")) as stored:
        bases = {
            name: torch.from_numpy(stored[name]).to(device()) for name in stored.files
        }
    network = Regression().to(device())
    network.load_state_dict(
        torch.load(artifact("text_regression"), map_location=device())
    )
    return network.eval(), bases


def mode_weights() -> torch.Tensor:
    weights = torch.zeros(len(VIEW_TYPES), len(MODES))
    for kind, (_, name) in enumerate(VIEW_TYPES):
        modes = VIEW_COMBINATIONS.get(name, (name,))
        weights[kind, [MODES.index(mode) for mode in modes]] = 1 / len(modes)
    return weights / len(VIEW_TYPES)


def mode_vectors(predictions: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "kgd,km->gmd", predictions, mode_weights().to(predictions.device)
    )


def predicted_vectors(
    network: Regression, bases: Bases, joint: torch.Tensor, galaxies: torch.Tensor
) -> np.ndarray:
    width = bases["components"].shape[2]
    found = np.empty((len(galaxies), len(MODES), width), dtype=np.float32)
    for start in range(0, len(galaxies), GALAXIES):
        rows = slice(start, start + GALAXIES)
        chosen = joint[galaxies[rows]]
        present = torch.ones(len(chosen), len(GRAM_NAMES), device=device())
        found[rows] = (
            mode_vectors(predicted(network, bases, chosen, present)).cpu().numpy()
        )
    return found


def generate_text_vectors() -> None:
    network, bases = stored_regression()
    joint = embeddings.loaded("joint_embedding", torch.float32)
    vectors = predicted_vectors(network, bases, joint, torch.arange(len(joint)))
    np.save(artifact("text_vectors"), vectors)
    np.save(artifact("text_query_mean"), embed_queries(generic_queries()).mean(axis=0))
