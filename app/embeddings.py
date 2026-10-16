from functools import cache
from typing import Annotated, NamedTuple, Self, get_args

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tqdm import tqdm

from . import pql
from .config import (
    GRAM_NAMES,
    MODES,
    TOKEN_EMBEDDING_RANK,
    GalaxyIndex,
    Mode,
    TokenMode,
    artifact,
    device,
)

BATCH = 1024
SEARCH_RANK = 256


class ModeQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    galaxy: GalaxyIndex
    modes: Annotated[tuple[Mode, ...], Field(min_length=1, max_length=len(MODES))]
    anywhere: Annotated[
        tuple[TokenMode, ...], Field(max_length=len(get_args(TokenMode)))
    ] = ()
    matches: Annotated[int, Field(ge=1, le=128)] = 32

    @model_validator(mode="after")
    def searches_anywhere_only_in_selected_modes(self) -> Self:
        if missing := set(self.anywhere) - set(self.modes):
            raise ValueError(
                f"any position names unselected {', '.join(sorted(missing))}"
            )
        return self


class ModeResults(NamedTuple):
    galaxies: np.ndarray
    scores: np.ndarray
    contributions: dict[str, np.ndarray]
    predicted: list[list[str]]


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


@cache
def stored(name: str) -> dict[str, torch.Tensor]:
    with np.load(artifact(f"token_embedding_{name}")) as found:
        return {
            part: torch.from_numpy(np.ascontiguousarray(found[part][:, :SEARCH_RANK]))
            for part in ("queries", "galleries")
        } | {"offsets": torch.from_numpy(found["offsets"])}


def mode_search(query: ModeQuery) -> ModeResults:
    modes = tuple(dict.fromkeys(query.modes))
    contributions = {}
    for mode in modes:
        found = stored(f"{mode}_anywhere" if mode in query.anywhere else mode)
        row = scores(found["queries"][[query.galaxy]], found, SEARCH_RANK)[0].double()
        contributions[mode] = (
            (row - row.mean()) / row.std(correction=0)
        ).numpy() / len(modes)
    scored = sum(contributions.values())
    order = np.argsort(-scored, kind="stable")
    galaxies = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return ModeResults(
        galaxies,
        scored[galaxies],
        {mode: values[galaxies] for mode, values in contributions.items()},
        pql.predicted(modes, galaxies),
    )


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
