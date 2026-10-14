import numpy as np
import torch

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


def centred(gram: torch.Tensor) -> torch.Tensor:
    symmetric = (gram + gram.T) / 2
    means = symmetric.mean(dim=0)
    return symmetric - means - means[:, None] + means.mean()


def embedding(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(centred(gram))
    eigenvalues, top = eigenvalues.flip(0), eigenvectors[:, -RANK:].flip(1)
    return top * eigenvalues[: top.shape[1]].clamp_min(0).sqrt(), eigenvalues


def scores(
    vectors: torch.Tensor, self_scores: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    return queries @ vectors.T - (vectors.square().sum(dim=1) - self_scores) / 2


def generate_token_embeddings() -> None:
    for name in GRAM_NAMES:
        gram = loaded(name)
        vectors, eigenvalues = embedding(gram)
        np.savez(
            artifact(f"token_embedding_{name}"),
            vectors=vectors.float().cpu().numpy(),
            self_scores=gram.diagonal().float().cpu().numpy(),
            eigenvalues=eigenvalues.cpu().numpy(),
        )
