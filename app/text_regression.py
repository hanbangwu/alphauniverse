import json
from collections.abc import Callable
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
    WEMM_DIM,
    artifact,
    device,
)
from .text_search import embed_queries, generic_queries

HIDDEN = 512
KIND_WIDTH = 16
BATCH = 512
MASKED = 0.3
DROPPED = 0.1
STEPS = 60
TRAINING_STEPS = 111_000
TEXT_VIEW_PARTS = 200
LOGGED = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DECAY = 0.999
SHARED_BASIS = 128
SCALE_FLOOR = 0.1
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


def views(path: Path | list[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = pq.read_table(path, columns=["galaxy", "view_type", "embedding"])
    embedding = table.column("embedding").combine_chunks().flatten().to_numpy()
    return (
        torch.tensor(table.column("galaxy").to_numpy(), device=device()).long(),
        torch.tensor(table.column("view_type").to_numpy(), device=device()).long(),
        torch.tensor(embedding.reshape(len(table), WEMM_DIM), device=device()).float(),
    )


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
        spread = pca.transform(found[fitted]).std(axis=0)
        scale.append(np.maximum(spread, SCALE_FLOOR * np.median(spread)))
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

    text_views.generate_text_views(TEXT_VIEW_PARTS)
    joint = embeddings.loaded("joint_embedding", torch.float32)
    paths = [text_views.part_path(index) for index in range(TEXT_VIEW_PARTS)]
    _, first_kinds, first_found = views(paths[:PARTS_PER_ROUND])
    bases = fitted_bases(first_kinds, first_found) | whitening(
        joint[text_views.training()]
    )
    del first_kinds, first_found
    held = evaluated(joint, bases)
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    count = TEXT_VIEW_PARTS * VIEWS_PER_PART
    owners = torch.empty(count, dtype=torch.long, device=device())
    kinds = torch.empty(count, dtype=torch.long, device=device())
    targets = torch.empty(count, TEXT_RANK, device=device())
    for index, path in enumerate(paths):
        rows = slice(index * VIEWS_PER_PART, (index + 1) * VIEWS_PER_PART)
        owners[rows], kinds[rows], found = views(path)
        targets[rows] = coefficients(
            bases, kinds[rows], residuals(bases["mean"], kinds[rows], found)
        )
    weights = training_weights(kinds)
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
            "steps": TRAINING_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "masked": MASKED,
            "dropped": DROPPED,
            "scale_floor": SCALE_FLOOR,
            "seed": SEED,
        },
    ) as run:
        losses = []
        for index in range(TRAINING_STEPS // STEPS):
            losses.append(stepped(network, average, optimiser, weights, inputs))
            if (index + 1) % LOGGED == 0:
                overall = identification(average.module, held, bases, everything)
                entry = {
                    "step": (index + 1) * STEPS,
                    "train/loss": float(np.mean(losses)),
                    "identification": overall.nanmean().item(),
                }
                run.log(entry)
                print(json.dumps(entry), flush=True)
                losses = []
        found = {"views": len(owners)} | report(average.module, held, bases)
        run.summary.update(found)
        print(json.dumps(found, indent=1))
    torch.save(average.module.state_dict(), artifact("text_regression"))
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
