import numpy as np
import torch
from tqdm import tqdm

from . import pql, templates
from .config import (
    GRAM_NAMES,
    MODES,
    TOKEN_EMBEDDING_RANK,
    TOKEN_GRAMS,
    artifact,
    device,
)

BATCH = 1024


def loaded(role: str) -> torch.Tensor:
    stored = np.load(artifact(role), mmap_mode="r")
    gram = torch.empty(stored.shape, dtype=torch.float64, device=device())
    for start in range(0, len(stored), BATCH):
        rows = slice(start, start + BATCH)
        gram[rows] = torch.tensor(stored[rows], device=gram.device)
    return gram


def centred(gram: torch.Tensor) -> torch.Tensor:
    return gram - gram.mean(dim=1, keepdim=True) - gram.mean(dim=0) + gram.mean()


def embedding(gram: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix = centred(gram)
    values, vectors = torch.linalg.eigh(matrix.T @ matrix)
    singular_values = values.flip(0).clamp_min(0).sqrt()
    rank = min(TOKEN_EMBEDDING_RANK, len(gram) - 1)
    right = vectors[:, -rank:].flip(1)
    scale = singular_values[:rank].sqrt()
    return {
        "queries": matrix @ right / scale,
        "galleries": right * scale,
        "offsets": gram.mean(dim=0),
        "self_scores": gram.diagonal(),
        "singular_values": singular_values,
    }


def saved(role: str, found: dict[str, torch.Tensor], **extra: np.ndarray) -> None:
    singular_values = found.pop("singular_values")
    np.savez(
        artifact(role),
        singular_values=singular_values.cpu().numpy(),
        **{part: values.float().cpu().numpy() for part, values in found.items()},
        **extra,
    )


def generate_token_embeddings() -> None:
    for name in tqdm(GRAM_NAMES, desc="token embeddings"):
        saved(f"token_embedding_{name}", embedding(loaded(f"token_gram_{name}")))


def predictive_widths() -> dict[str, np.ndarray]:
    with np.load(artifact("observed_reference")) as references:
        depths = {mode: references[f"{mode}_depth"] for mode in MODES}
    count = pql.predictions().num_rows
    found = {mode: np.empty(count, np.float32) for mode in MODES}
    start = 0
    for rows in tqdm(pql.predictions().to_batches(), desc="widths"):
        for mode, values in templates.widths(rows, depths).items():
            found[mode][start : start + rows.num_rows] = values
        start += rows.num_rows
    return found


def generate_template_embeddings() -> None:
    widths = predictive_widths()
    for mode, names in TOKEN_GRAMS.items():
        for name in tqdm(names, desc=mode):
            saved(
                f"template_embedding_{name}",
                embedding(loaded(f"template_gram_{name}")),
                widths=widths[mode],
            )
