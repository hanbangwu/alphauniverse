import json
import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import jacfwd, vmap
from torch.optim.swa_utils import AveragedModel

from . import text_views
from .config import (
    DATASET_REVISION,
    GEMMA_DIM,
    GRAM_NAMES,
    MODES,
    SEED,
    TOKEN_GRAMS,
    VIEW_TYPES,
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


class Velocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kind = nn.Embedding(len(VIEW_TYPES), KIND_WIDTH)
        width = RANK + CONDITION + len(GRAM_NAMES) + KIND_WIDTH + 2 * FREQUENCIES
        self.layers = nn.Sequential(
            nn.Linear(width, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, RANK),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        present: torch.Tensor,
        kind: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(
            torch.cat(
                [x, condition, present, self.kind(kind), time_features(t)], dim=-1
            )
        )


class Reverse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(CONDITION, HIDDEN)
        self.context_layer = nn.Linear(
            GEMMA_DIM + 1 + 2 * FREQUENCIES, HIDDEN, bias=False
        )
        self.tail = nn.Sequential(
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, CONDITION),
        )

    def context(
        self, t: torch.Tensor, view: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        return self.context_layer(
            torch.cat([view * present, present, time_features(t)], dim=-1)
        )

    def velocity(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.tail(self.first(x) + context)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        view: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(x, self.context(t, view, present))


def stored(index: int) -> Path:
    path = artifact("text_views") / f"part-{index:05d}.parquet"
    if not path.exists():
        per_round = text_views.ROUND // text_views.PART
        text_views.generate_text_views((index // per_round + 1) * per_round)
    return path


def views(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["galaxy", "view_type", "embedding"])
    embeddings = table.column("embedding").combine_chunks().flatten().to_numpy()
    return (
        table.column("galaxy").to_numpy().astype(np.int64),
        table.column("view_type").to_numpy().astype(np.int64),
        embeddings.astype(np.float32).reshape(len(table), GEMMA_DIM),
    )


def fitted_bases(view_types: np.ndarray, embeddings: np.ndarray) -> Bases:
    from sklearn.decomposition import PCA

    mean = np.stack(
        [embeddings[view_types == kind].mean(axis=0) for kind in range(len(VIEW_TYPES))]
    )
    residuals = embeddings - mean[view_types]
    residuals /= np.linalg.norm(residuals, axis=1, keepdims=True)
    components, centre, scale = [], [], []
    for kind, (_, name) in enumerate(VIEW_TYPES):
        fitted = view_types == kind
        if fitted.sum() < SHARED_BASIS:
            siblings = [
                other for other, view in enumerate(VIEW_TYPES) if view[1] == name
            ]
            fitted = np.isin(view_types, siblings)
        pca = PCA(RANK, random_state=SEED).fit(residuals[fitted])
        components.append(pca.components_)
        centre.append(pca.mean_)
        scale.append(pca.transform(residuals[fitted]).std(axis=0))
    found = {
        "mean": mean,
        "components": np.stack(components),
        "centre": np.stack(centre),
        "scale": np.stack(scale),
    }
    return {name: torch.from_numpy(value).to(device()) for name, value in found.items()}


def residuals(
    bases: Bases, view_types: torch.Tensor, embeddings: torch.Tensor
) -> torch.Tensor:
    return F.normalize(embeddings - bases["mean"][view_types], dim=-1)


def coefficients(
    bases: Bases, view_types: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    found = torch.empty(len(directions), RANK, device=directions.device)
    for kind in view_types.unique():
        chosen = view_types == kind
        centred = directions[chosen] - bases["centre"][kind]
        found[chosen] = centred @ bases["components"][kind].T / bases["scale"][kind]
    return found


def whitening(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from sklearn.decomposition import PCA

    pca = PCA(CONDITION, whiten=True, random_state=SEED).fit(rows.cpu().numpy())
    projection = pca.components_.T / np.sqrt(pca.explained_variance_)
    return (
        torch.from_numpy(pca.mean_).float().to(rows.device),
        torch.from_numpy(projection).float().to(rows.device),
    )


def condition(
    rows: torch.Tensor,
    present: torch.Tensor,
    offset: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    masked = rows.unflatten(1, (len(GRAM_NAMES), -1)) * present[..., None]
    return (masked.flatten(1) - offset) @ projection


def masks(count: int) -> torch.Tensor:
    present = (torch.rand(count, len(GRAM_NAMES), device=device()) > MASKED).float()
    return present * (torch.rand(count, 1, device=device()) > DROPPED)


def averaged(network: nn.Module) -> AveragedModel:
    return AveragedModel(
        network,
        avg_fn=lambda mean, current, count: (
            mean
            + (current - mean)
            * (1 - torch.clamp((1 + count) / (10 + count), max=DECAY))
        ),
    )


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


def identification(
    clouds: torch.Tensor, held: HeldOut, bases: Bases, kappas: tuple[float, ...]
) -> torch.Tensor:
    kappa = torch.tensor(kappas, device=clouds.device)[:, None, None, None]
    own = torch.searchsorted(held.galaxies, held.owners)
    gains = torch.full((len(kappas), len(VIEW_TYPES)), torch.nan, device=clouds.device)
    for kind in held.view_types.unique():
        chosen = held.view_types == kind
        found = cosines(
            clouds[:, kind],
            held.directions[chosen],
            bases["components"][kind],
            bases["centre"][kind],
        )
        scores = torch.logsumexp(kappa * found, dim=-1) - math.log(POINTS)
        ranked = scores[:, torch.arange(len(found), device=found.device), own[chosen]]
        gains[:, kind] = (ranked - torch.logsumexp(scores, dim=-1)).mean(dim=1)
    return gains + math.log(len(held.galaxies))


def evaluated(joint: torch.Tensor, bases: Bases) -> HeldOut:
    owners, view_types, embeddings = (
        torch.from_numpy(found).to(device())
        for found in views(artifact("text_views_held_out"))
    )
    galaxies = owners.unique()
    return HeldOut(
        galaxies,
        joint[galaxies],
        owners,
        view_types,
        residuals(bases, view_types, embeddings),
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


def gains(
    network: Velocity,
    held: HeldOut,
    bases: Bases,
    whitened: tuple[torch.Tensor, torch.Tensor],
    kappas: tuple[float, ...],
    present: torch.Tensor,
) -> torch.Tensor:
    present = present.expand(len(held.galaxies), -1)
    found = clouds(
        network, condition(held.rows, present, *whitened), present, bases["scale"]
    )
    return identification(found, held, bases, kappas)


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
    network: Reverse,
    points: torch.Tensor,
    embedded: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    x = points
    divergence = torch.zeros(len(x), device=x.device)
    jacobian = vmap(
        jacfwd(lambda row, context: network.velocity(row[None], context[None])[0])
    )
    for step in range(EULER, 0, -1):
        t = torch.full((len(x), 1), step / EULER, device=x.device)
        context = network.context(t, embedded, present)
        divergence += jacobian(x, context).diagonal(dim1=-2, dim2=-1).sum(-1) / EULER
        x = x - network.velocity(x, context) / EULER
    normal = -0.5 * (x**2).sum(-1) - 0.5 * CONDITION * math.log(2 * math.pi)
    return normal - divergence


def reverse_trained(
    prefixes: list[int],
    owners: torch.Tensor,
    contexts: torch.Tensor,
    targets: torch.Tensor,
) -> Reverse:
    torch.manual_seed(SEED)
    network = Reverse().to(device())
    average = averaged(network)
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def inputs(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        present = (torch.rand(BATCH, 1, device=device()) > DROPPED).float()
        view = contexts[batch].float() * math.sqrt(GEMMA_DIM)
        return targets[owners[batch]], view, present

    for count in prefixes:
        stepped(network, average, optimiser, count, inputs)
    return average.module


def reverse_identification(
    network: Reverse, held: HeldOut, targets: torch.Tensor
) -> torch.Tensor:
    count = len(held.galaxies)
    prior = log_density(
        network,
        targets,
        torch.zeros(count, GEMMA_DIM, device=device()),
        torch.zeros(count, 1, device=device()),
    )
    contexts = held.directions * math.sqrt(GEMMA_DIM)
    ratios = torch.empty(len(contexts), count, device=device())
    per = max(1, PAIRS // count)
    for start in range(0, len(contexts), per):
        chunk = contexts[start : start + per]
        found = log_density(
            network,
            targets.repeat(len(chunk), 1),
            chunk.repeat_interleave(count, dim=0),
            torch.ones(len(chunk) * count, 1, device=device()),
        )
        ratios[start : start + per] = found.view(len(chunk), count) - prior
    rows = torch.arange(len(ratios), device=ratios.device)
    own = ratios[rows, torch.searchsorted(held.galaxies, held.owners)]
    gain = own - torch.logsumexp(ratios, dim=-1) + math.log(count)
    found = torch.full((len(VIEW_TYPES),), torch.nan, device=device())
    for kind in held.view_types.unique():
        found[kind] = gain[held.view_types == kind].mean()
    return found


def per_type(found: torch.Tensor) -> dict[str, float]:
    values = found.cpu().numpy()
    return {"mean": round(float(np.nanmean(values)), 4)} | {
        f"{kind} {name}": round(float(value), 4)
        for (kind, name), value in zip(VIEW_TYPES, values, strict=True)
        if np.isfinite(value)
    }


def train_text_flow() -> None:
    import wandb

    joint = torch.from_numpy(np.load(artifact("joint_embedding"))).to(device())
    training = np.setdiff1d(np.arange(len(joint)), text_views.held_out())
    whitened = whitening(joint[training])
    per_round = text_views.ROUND // text_views.PART
    view_types, embeddings = (
        np.concatenate(found)
        for found in zip(*(views(stored(index))[1:] for index in range(per_round)))
    )
    bases = fitted_bases(view_types, embeddings)
    held = evaluated(joint, bases)
    everything = torch.ones(1, len(GRAM_NAMES), device=device())
    targets = torch.empty(CAP, RANK, device=device())
    owners = torch.empty(CAP, dtype=torch.long, device=device())
    kinds = torch.empty(CAP, dtype=torch.long, device=device())
    contexts = torch.empty(CAP, GEMMA_DIM, dtype=torch.float16, device=device())
    torch.manual_seed(SEED)
    network = Velocity().to(device())
    average = averaged(network)
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def inputs(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        present = masks(BATCH)
        rows = joint[owners[batch]]
        return (
            targets[batch],
            condition(rows, present, *whitened),
            present,
            kinds[batch],
        )

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
        for index in range(CAP // text_views.PART):
            galaxy, view_type, embedding = (
                torch.from_numpy(found).to(device()) for found in views(stored(index))
            )
            rows = slice(filled, filled + len(galaxy))
            directions = residuals(bases, view_type, embedding)
            targets[rows] = coefficients(bases, view_type, directions)
            owners[rows], kinds[rows], contexts[rows] = galaxy, view_type, directions
            losses = []
            for start in range(filled, rows.stop, CHUNK):
                prefixes.append(min(start + CHUNK, rows.stop))
                losses.append(
                    stepped(network, average, optimiser, prefixes[-1], inputs)
                )
            filled = rows.stop
            found = gains(average.module, held, bases, whitened, KAPPAS, everything)
            overall = found.nanmean(dim=1)
            run.log(
                {"views": filled, "train/loss": float(np.mean(losses))}
                | {
                    f"identification/kappa {kappa:g}": value
                    for kappa, value in zip(KAPPAS, overall.tolist(), strict=True)
                }
            )
            if overall.max().item() > best + IMPROVEMENT:
                best, best_part, best_gains = overall.max().item(), index, found
                state, seen = deepcopy(average.module.state_dict()), len(prefixes)
            if index - best_part >= PATIENCE:
                break
        network.load_state_dict(state)
        network.eval()
        kappa = KAPPAS[int(best_gains.nanmean(dim=1).argmax())]
        prototype = subset(
            held, torch.from_numpy(text_views.PROTOTYPE_HELD_OUT).to(device())
        )
        reverse = reverse_trained(
            prefixes[:seen],
            owners,
            contexts,
            condition(joint, everything, *whitened),
        )
        report = {
            "views": prefixes[seen - 1],
            "kappa": kappa,
            "identification": {
                f"kappa {value:g}": per_type(row)
                for value, row in zip(KAPPAS, best_gains, strict=True)
            },
            "prototype identification": {
                f"kappa {value:g}": per_type(row)
                for value, row in zip(
                    KAPPAS,
                    gains(network, prototype, bases, whitened, KAPPAS, everything),
                    strict=True,
                )
            },
            "masked": {
                name: per_type(
                    gains(network, held, bases, whitened, (kappa,), present[None])[0]
                )
                for name, present in patterns().items()
            },
            "reverse": per_type(
                reverse_identification(
                    reverse, held, condition(held.rows, everything, *whitened)
                )
            ),
        }
        run.summary.update(report)
        print(json.dumps(report, indent=1))
    torch.save(state, artifact("text_flow"))
    np.savez(
        artifact("text_bases"),
        **{name: value.cpu().numpy() for name, value in bases.items()},
        offset=whitened[0].cpu().numpy(),
        projection=whitened[1].cpu().numpy(),
        kappa=np.float32(kappa),
    )
