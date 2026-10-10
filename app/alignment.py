import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from .config import (
    DATASET_REVISION,
    DIM,
    GEMMA,
    GEMMA_DIM,
    SEED,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    artifact,
    device,
)

BATCH = 1024
EPOCHS = 50
HIDDEN = 2048
LEARNING_RATE = 1e-3
RECALL_AT = 10
VALIDATION = 0.3


class AlignmentMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, GEMMA_DIM)
        )

    def forward(self, aion: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(aion), dim=-1)


def pairs() -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    table = pq.read_table(artifact("pairs"), columns=["galaxy", "aion", "gemma"])
    galaxy = np.asarray(table.column("galaxy"))
    aion, gemma = (
        torch.as_tensor(
            np.asarray(table.column(name).combine_chunks().flatten()).reshape(
                len(galaxy), -1
            ),
            device=device(),
        )
        for name in ("aion", "gemma")
    )
    return galaxy, aion, gemma


def held_out(galaxy: np.ndarray) -> torch.Tensor:
    _, validation = train_test_split(galaxy, test_size=VALIDATION, random_state=SEED)
    return torch.as_tensor(np.isin(galaxy, validation), device=device())


def fit_linear(aion: torch.Tensor, gemma: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(F.pad(aion, (0, 1), value=1.0), gemma).solution


def apply_linear(weights: torch.Tensor, aion: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.pad(aion, (0, 1), value=1.0) @ weights, dim=-1)


def recall(predicted: torch.Tensor, target: torch.Tensor) -> float:
    found = 0
    for start in range(0, len(predicted), BATCH):
        scores = predicted[start : start + BATCH] @ target.T
        own = scores[
            torch.arange(len(scores), device=device()),
            torch.arange(start, start + len(scores), device=device()),
        ]
        found += int(((scores > own[:, None]).sum(dim=1) < RECALL_AT).sum())
    return found / len(predicted)


def metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "cosine": F.cosine_similarity(predicted, target).mean().item(),
        f"recall@{RECALL_AT}": recall(predicted, target),
    }


def generate_alignment() -> None:
    import wandb

    galaxy, aion, gemma = pairs()
    validation = held_out(galaxy)
    linear = fit_linear(aion[~validation], gemma[~validation])

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="train",
        tags=["alignment"],
        group=DATASET_REVISION,
        mode=WANDB_MODE,
        config={
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "hidden": HIDDEN,
            "validation": VALIDATION,
            "seed": SEED,
            "target": GEMMA,
        },
    ) as run:
        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch", summary="min")
        run.define_metric("validation/*", step_metric="epoch", summary="max")
        run.summary.update(
            {
                f"linear/{name}": value
                for name, value in metrics(
                    apply_linear(linear, aion[validation]), gemma[validation]
                ).items()
            }
        )
        torch.manual_seed(SEED)
        model = AlignmentMap().to(device())
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        training = torch.nonzero(~validation).squeeze(1)

        for epoch in range(EPOCHS):
            losses = []
            for batch in tqdm(
                training[torch.randperm(len(training), device=device())].split(BATCH),
                desc="alignment",
            ):
                step = 1 - F.cosine_similarity(model(aion[batch]), gemma[batch]).mean()
                optimizer.zero_grad()
                step.backward()
                optimizer.step()
                losses.append(step.detach())

            with torch.no_grad():
                scores = metrics(model(aion[validation]), gemma[validation])
            run.log(
                {"epoch": epoch, "train/loss": torch.stack(losses).mean().item()}
                | {f"validation/{name}": value for name, value in scores.items()}
            )

    torch.save({"linear": linear, "mlp": model.state_dict()}, artifact("alignment"))


def generate_aion_gemma_space() -> None:
    _, aion, _ = pairs()
    model = AlignmentMap().to(device())
    model.load_state_dict(
        torch.load(artifact("alignment"), map_location=device())["mlp"]
    )
    with torch.no_grad():
        vectors = model.eval()(aion).cpu().numpy()
    np.save(artifact("aion_gemma_space"), vectors)
