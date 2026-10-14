import numpy as np
import torch
from tqdm import tqdm

from .config import GRAM_NAMES, artifact, device

RANK = 1024
BATCH = 1024


def loaded(name: str) -> torch.Tensor:
    stored = np.load(artifact(f"token_gram_{name}"), mmap_mode="r")
    gram = torch.empty(stored.shape, dtype=torch.float64, device=device())
    for start in range(0, len(stored), BATCH):
        gram[start : start + BATCH] = torch.from_numpy(
            np.array(stored[start : start + BATCH])
        )
    return gram


def centred(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = gram.mean(dim=0)
    return gram - gram.mean(dim=1, keepdim=True) - offsets + gram.mean(), offsets


def embedding(gram: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix, offsets = centred(gram)
    values, vectors = torch.linalg.eigh(matrix.T @ matrix)
    singular_values = values.flip(0).clamp_min(0).sqrt()
    right = vectors[:, -min(RANK, len(gram) - 1) :].flip(1)
    scale = singular_values[: right.shape[1]].sqrt()
    return {
        "queries": matrix @ right / scale,
        "galleries": right * scale,
        "offsets": offsets,
        "singular_values": singular_values,
    }


def scores(
    queries: torch.Tensor, galleries: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    return queries @ galleries.T + offsets


def generate_token_embeddings() -> None:
    for name in tqdm(GRAM_NAMES, desc="token embeddings"):
        found = embedding(loaded(name))
        np.savez(
            artifact(f"token_embedding_{name}"),
            singular_values=found.pop("singular_values").cpu().numpy(),
            **{part: values.float().cpu().numpy() for part, values in found.items()},
        )
