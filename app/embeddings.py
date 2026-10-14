import numpy as np
import torch
from tqdm import tqdm

from .config import GRAM_NAMES, TOKEN_EMBEDDING_RANK, artifact, device

BATCH = 1024


def loaded(role: str, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    stored = np.load(artifact(role), mmap_mode="r")
    gram = torch.empty(stored.shape, dtype=dtype, device=device())
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


def query_vectors(rows: torch.Tensor, found: dict[str, torch.Tensor]) -> torch.Tensor:
    offsets, galleries = found["offsets"], found["galleries"]
    matrix = rows - rows.mean(dim=1, keepdim=True) - offsets + offsets.mean()
    singular_values = found["singular_values"][: galleries.shape[1]].to(galleries)
    return matrix @ (galleries / singular_values)


def scores(
    queries: torch.Tensor, found: dict[str, torch.Tensor], rank: int
) -> torch.Tensor:
    return queries[:, :rank] @ found["galleries"][:, :rank].T + found["offsets"]


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
