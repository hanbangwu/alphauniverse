from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Annotated, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from core.artifacts import artifact
from core.text_model import BATCH, LOADING


class TextQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1, max_length=500)]
    matches: Annotated[int, Field(ge=1, le=128)] = 32


class TextVectors(NamedTuple):
    modes: np.ndarray
    vectors: torch.Tensor
    baseline: torch.Tensor
    query_mean: torch.Tensor


class TextResults(NamedTuple):
    galaxies: torch.Tensor
    scores: torch.Tensor
    contributions: torch.Tensor


@cache
def text_vectors() -> TextVectors:
    modes = np.load(artifact("text_vectors"), mmap_mode="r")
    vectors = torch.empty(len(modes), modes.shape[2])
    baseline = torch.zeros(modes.shape[1:])
    for start in range(0, len(modes), BATCH):
        rows = torch.from_numpy(np.array(modes[start : start + BATCH]))
        vectors[start : start + BATCH] = rows.sum(dim=1)
        baseline += rows.sum(dim=0)
    query_mean = torch.from_numpy(np.load(artifact("text_query_mean")))
    return TextVectors(modes, vectors, baseline / len(modes), query_mean)


def query_directions(embedded: np.ndarray, mean: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.from_numpy(embedded) - mean, dim=-1)


def text_search(
    query: TextQuery, embed: Callable[[list[str]], np.ndarray]
) -> TextResults:
    with LOADING:
        stored = text_vectors()
    embedded = embed([query.text])[0]
    direction = query_directions(embedded, stored.query_mean)
    scores = stored.vectors @ direction
    top = scores.topk(min(query.matches, len(scores))).indices
    modes = torch.from_numpy(stored.modes[top.numpy()])
    return TextResults(top, scores[top], (modes - stored.baseline) @ direction)
