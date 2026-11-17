from __future__ import annotations

from collections.abc import Callable
from functools import cache
from threading import Lock
from typing import TYPE_CHECKING, Annotated, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

from app.config import (
    QUERY_PREFIX,
    WEMM,
    WEMM_DIM,
    artifact,
    device,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BATCH = 1024
LOADING = Lock()


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
def text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    kernels = (
        {
            "dtype": torch.bfloat16,
            "use_kernels": True,
            "attn_implementation": "kernels-community/flash-attn2",
        }
        if device().type == "cuda"
        else {}
    )
    model = SentenceTransformer(
        WEMM,
        device=device(),
        trust_remote_code=True,
        truncate_dim=WEMM_DIM,
        model_kwargs=kernels,
    )
    trained = Tokenizer.from_file(hf_hub_download(WEMM, "tokenizer.json"))
    model[0].processor.tokenizer.backend_tokenizer.normalizer = trained.normalizer
    return model


def augmented(text: str) -> str:
    return QUERY_PREFIX.format(text)


def embed_queries(texts: list[str]) -> np.ndarray:
    with LOADING:
        model = text_model()
    return model.encode_query(
        [augmented(text) for text in texts], normalize_embeddings=True
    )


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
