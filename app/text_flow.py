import json
import math
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.func import jacfwd, vmap
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
    TOKEN_GRAMS,
    VIEW_TYPES,
    VIEWS_PER_PART,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    artifact,
    device,
)

RANK = 32
CONDITION = 64
HIDDEN = 256
KIND_WIDTH = 16
FREQUENCIES = 8
BATCH = 512
MASKED = 0.3
DROPPED = 0.1
CHUNK = 400
STEPS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DECAY = 0.999
PATIENCE = 5
IMPROVEMENT = 0.01
POINTS = 16
EULER = 32
KAPPAS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
BETAS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
SHARED_BASIS = 128
CAP = 1_000_000
PAIRS = 8192

Bases = dict[str, torch.Tensor]


class HeldOut(NamedTuple):
    galaxies: torch.Tensor
    rows: torch.Tensor
    owners: torch.Tensor
    view_types: torch.Tensor
    directions: torch.Tensor


def time_features(t: torch.Tensor) -> torch.Tensor:
    angles = t * 2.0 ** torch.arange(FREQUENCIES, device=t.device) * torch.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class Flow(nn.Module):
    def __init__(self, target: int, conditions: int) -> None:
        super().__init__()
        self.first = nn.Linear(target, HIDDEN)
        self.context_layer = nn.Linear(conditions + 2 * FREQUENCIES, HIDDEN, bias=False)
        self.tail = nn.Sequential(
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, target),
        )

    def context(self, t: torch.Tensor, *conditions: torch.Tensor) -> torch.Tensor:
        return self.context_layer(torch.cat([*conditions, time_features(t)], dim=-1))

    def velocity(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.tail(self.first(x) + context)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *conditions: torch.Tensor
    ) -> torch.Tensor:
        return self.velocity(x, self.context(t, *conditions))


class Velocity(Flow):
    def __init__(self) -> None:
        super().__init__(RANK, CONDITION + len(GRAM_NAMES) + KIND_WIDTH)
        self.kind = nn.Embedding(len(VIEW_TYPES), KIND_WIDTH)

    def context(
        self,
        t: torch.Tensor,
        condition: torch.Tensor,
        present: torch.Tensor,
        kind: torch.Tensor,
    ) -> torch.Tensor:
        return super().context(t, condition, present, self.kind(kind))


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
        pca = PCA(RANK, random_state=SEED).fit(found[fitted])
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
    pca = PCA(CONDITION, whiten=True, random_state=SEED).fit(rows.cpu().numpy())
    projection = pca.components_.T / np.sqrt(pca.explained_variance_)
    return {
        "offset": torch.from_numpy(pca.mean_).to(device()),
        "projection": torch.from_numpy(projection).to(device()),
    }


def coefficients(
    bases: Bases, view_types: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    found = torch.empty(len(directions), RANK, device=directions.device)
    for kind in view_types.unique():
        chosen = view_types == kind
        centred = directions[chosen] - bases["centre"][kind]
        found[chosen] = centred @ bases["components"][kind].T / bases["scale"][kind]
    return found


def condition(rows: torch.Tensor, present: torch.Tensor, bases: Bases) -> torch.Tensor:
    masked = rows.unflatten(1, (len(GRAM_NAMES), -1)) * present[..., None]
    return (masked.flatten(1) - bases["offset"]) @ bases["projection"]


def masks(count: int) -> torch.Tensor:
    present = (torch.rand(count, len(GRAM_NAMES), device=device()) > MASKED).float()
    return present * (torch.rand(count, 1, device=device()) > DROPPED)


def decayed(
    averages: list[torch.Tensor], currents: list[torch.Tensor], count: torch.Tensor
) -> None:
    weight = 1 - torch.clamp((1 + count) / (10 + count), max=DECAY)
    for average, current in zip(averages, currents, strict=True):
        average.lerp_(current, weight)


def trainable(
    build: Callable[[], nn.Module],
) -> tuple[nn.Module, AveragedModel, torch.optim.Optimizer]:
    torch.manual_seed(SEED)
    network = build().to(device())
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    return network, AveragedModel(network, multi_avg_fn=decayed), optimiser


def flow_loss(
    network: nn.Module, target: torch.Tensor, *conditions: torch.Tensor
) -> torch.Tensor:
    start = torch.randn_like(target)
    t = torch.rand(len(target), 1, device=target.device)
    velocity = network((1 - t) * start + t * target, t, *conditions)
    return ((velocity - (target - start)) ** 2).mean()


def stepped(
    network: nn.Module,
    average: AveragedModel,
    optimiser: torch.optim.Optimizer,
    count: int,
    inputs: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
) -> float:
    losses = []
    for _ in range(STEPS):
        loss = flow_loss(
            network, *inputs(torch.randint(count, (BATCH,), device=device()))
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        average.update_parameters(network)
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


@torch.no_grad()
def clouds(
    network: Velocity,
    conditions: torch.Tensor,
    present: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    kinds = len(VIEW_TYPES)
    generator = torch.Generator(conditions.device).manual_seed(SEED)
    kind = torch.arange(kinds, device=conditions.device).repeat_interleave(POINTS)
    kind = kind.repeat(len(conditions))
    conditions = conditions.repeat_interleave(kinds * POINTS, dim=0)
    present = present.repeat_interleave(kinds * POINTS, dim=0)
    x = torch.randn(len(kind), RANK, generator=generator, device=conditions.device)
    for step in range(EULER):
        t = torch.full((len(x), 1), step / EULER, device=x.device)
        x = x + network(x, t, conditions, present, kind) / EULER
    return x.view(-1, kinds, POINTS, RANK) * scale[:, None]


def cosines(
    points: torch.Tensor,
    directions: torch.Tensor,
    components: torch.Tensor,
    centre: torch.Tensor,
) -> torch.Tensor:
    squared = (points**2).sum(-1) + 2 * points @ (components @ centre) + centre @ centre
    along = torch.einsum("gpr,vr->vgp", points, directions @ components.T)
    return (along + (directions @ centre)[:, None, None]) / squared.sqrt()


def cloud_scores(clouds: torch.Tensor, held: HeldOut, bases: Bases) -> torch.Tensor:
    kappa = torch.tensor(KAPPAS, device=clouds.device)[:, None, None, None]
    scores = torch.empty(
        len(KAPPAS), len(held.owners), len(held.galaxies), device=clouds.device
    )
    for kind in held.view_types.unique():
        chosen = held.view_types == kind
        found = cosines(
            clouds[:, kind],
            held.directions[chosen],
            bases["components"][kind],
            bases["centre"][kind],
        )
        scores[:, chosen] = torch.logsumexp(kappa * found, dim=-1)
    return scores


def gains(scores: torch.Tensor, held: HeldOut) -> torch.Tensor:
    own = torch.searchsorted(held.galaxies, held.owners)
    rows = torch.arange(len(own), device=own.device)
    found = scores.log_softmax(dim=-1)[..., rows, own] + math.log(len(held.galaxies))
    sums = found.new_zeros(*found.shape[:-1], len(VIEW_TYPES))
    sums.index_add_(-1, held.view_types, found)
    return sums / torch.bincount(held.view_types, minlength=len(VIEW_TYPES))


def identification(
    network: Velocity, held: HeldOut, bases: Bases, present: torch.Tensor
) -> torch.Tensor:
    present = present.expand(len(held.galaxies), -1)
    found = clouds(
        network, condition(held.rows, present, bases), present, bases["scale"]
    )
    return gains(cloud_scores(found, held, bases), held)


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


@torch.no_grad()
def log_density(
    network: Flow, points: torch.Tensor, *conditions: torch.Tensor
) -> torch.Tensor:
    x = points
    divergence = torch.zeros(len(x), device=x.device)
    jacobian = vmap(jacfwd(network.velocity))
    for step in range(EULER, 0, -1):
        t = torch.full((len(x), 1), step / EULER, device=x.device)
        context = network.context(t, *conditions)
        divergence += jacobian(x, context).diagonal(dim1=-2, dim2=-1).sum(-1) / EULER
        x = x - network.velocity(x, context) / EULER
    normal = -0.5 * (x**2).sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi)
    return normal - divergence


def reverse_conditions(
    directions: torch.Tensor, present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return directions * math.sqrt(GEMMA_DIM) * present, present


def reverse_trained(
    prefixes: list[int],
    owners: torch.Tensor,
    contexts: torch.Tensor,
    targets: torch.Tensor,
) -> Flow:
    network, average, optimiser = trainable(partial(Flow, CONDITION, GEMMA_DIM + 1))

    def inputs(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        present = (torch.rand(BATCH, 1, device=device()) > DROPPED).float()
        found = reverse_conditions(contexts[batch].float(), present)
        return targets[owners[batch]], *found

    for count in prefixes:
        stepped(network, average, optimiser, count, inputs)
    return average.module


def reverse_identification(
    network: Flow, held: HeldOut, targets: torch.Tensor
) -> torch.Tensor:
    count = len(held.galaxies)
    absent = torch.zeros(count, 1, device=device())
    prior = log_density(
        network,
        targets,
        *reverse_conditions(torch.zeros(count, GEMMA_DIM, device=device()), absent),
    )
    ratios = torch.empty(len(held.owners), count, device=device())
    per = PAIRS // count
    for start in range(0, len(held.owners), per):
        chunk = held.directions[start : start + per]
        present = torch.ones(len(chunk) * count, 1, device=device())
        found = log_density(
            network,
            targets.repeat(len(chunk), 1),
            *reverse_conditions(chunk.repeat_interleave(count, dim=0), present),
        )
        ratios[start : start + per] = found.view(len(chunk), count) - prior
    beta = torch.tensor(BETAS, device=device())[:, None, None]
    return gains(beta * ratios, held)


def per_type(found: torch.Tensor) -> dict[str, float]:
    values = found.cpu().numpy()
    return {"mean": round(float(np.nanmean(values)), 4)} | {
        f"{kind} {name}": round(float(value), 4)
        for (kind, name), value in zip(VIEW_TYPES, values, strict=True)
        if np.isfinite(value)
    }


def labelled(
    name: str, values: tuple[float, ...], found: torch.Tensor
) -> dict[str, dict[str, float]]:
    return {
        f"{name} {value:g}": per_type(row)
        for value, row in zip(values, found, strict=True)
    }


def report(
    flow: Velocity, measured: torch.Tensor, held: HeldOut, bases: Bases, reverse: Flow
) -> dict:
    choice = int(measured.nanmean(dim=1).argmax())
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    prototype = subset(held, torch.from_numpy(PROTOTYPE_HELD_OUT).to(device()))
    return {
        "kappa": KAPPAS[choice],
        "identification": labelled("kappa", KAPPAS, measured),
        "prototype identification": labelled(
            "kappa", KAPPAS, identification(flow, prototype, bases, everything)
        ),
        "masked": {
            name: per_type(identification(flow, held, bases, present[None])[choice])
            for name, present in patterns().items()
        },
        "reverse": labelled(
            "beta",
            BETAS,
            reverse_identification(
                reverse, held, condition(held.rows, everything, bases)
            ),
        ),
    }


def train_text_flow() -> None:
    import wandb

    text_views.generate_text_views(PARTS_PER_ROUND)
    joint = embeddings.loaded("joint_embedding", torch.float32)
    bases = fitted_bases(*first_round()) | whitening(joint[text_views.training()])
    held = evaluated(joint, bases)
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    targets = torch.empty(CAP, RANK, device=device())
    owners = torch.empty(CAP, dtype=torch.long, device=device())
    kinds = torch.empty(CAP, dtype=torch.long, device=device())
    contexts = torch.empty(CAP, GEMMA_DIM, dtype=torch.float16, device=device())
    network, average, optimiser = trainable(Velocity)

    def inputs(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        present = masks(BATCH)
        rows = joint[owners[batch]]
        return targets[batch], condition(rows, present, bases), present, kinds[batch]

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="train",
        tags=["text_flow"],
        group=DATASET_REVISION,
        mode=WANDB_MODE,
        config={
            "rank": RANK,
            "condition": CONDITION,
            "hidden": HIDDEN,
            "batch": BATCH,
            "chunk": CHUNK,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "masked": MASKED,
            "dropped": DROPPED,
            "points": POINTS,
            "euler": EULER,
            "seed": SEED,
        },
    ) as run:
        prefixes, filled, best, best_part = [], 0, -math.inf, 0
        for index in range(CAP // VIEWS_PER_PART):
            if not text_views.part_path(index).exists():
                rounds = index // PARTS_PER_ROUND + 1
                text_views.generate_text_views(rounds * PARTS_PER_ROUND)
            galaxy, view_type, embedding = views(text_views.part_path(index))
            rows = slice(filled, filled + len(galaxy))
            directions = residuals(bases["mean"], view_type, embedding)
            targets[rows] = coefficients(bases, view_type, directions)
            owners[rows], kinds[rows], contexts[rows] = galaxy, view_type, directions
            losses = []
            for start in range(filled, rows.stop, CHUNK):
                prefixes.append(min(start + CHUNK, rows.stop))
                losses.append(
                    stepped(network, average, optimiser, prefixes[-1], inputs)
                )
            filled = rows.stop
            measured = identification(average.module, held, bases, everything)
            overall = measured.nanmean(dim=1)
            entry = {"views": filled, "train/loss": float(np.mean(losses))} | {
                f"identification/kappa {kappa:g}": value
                for kappa, value in zip(KAPPAS, overall.tolist(), strict=True)
            }
            run.log(entry)
            print(json.dumps(entry), flush=True)
            if overall.max().item() > best + IMPROVEMENT:
                best, best_part, best_measured = overall.max().item(), index, measured
                best_flow, seen = deepcopy(average.module), len(prefixes)
            if index - best_part >= PATIENCE:
                break
        reverse = reverse_trained(
            prefixes[:seen], owners, contexts, condition(joint, everything, bases)
        )
        found = {"views": prefixes[seen - 1]} | report(
            best_flow, best_measured, held, bases, reverse
        )
        run.summary.update(found)
        print(json.dumps(found, indent=1))
    torch.save(best_flow.state_dict(), artifact("text_flow"))
    np.savez(
        artifact("text_bases"),
        **{name: value.cpu().numpy() for name, value in bases.items()},
        kappa=np.float32(found["kappa"]),
    )
