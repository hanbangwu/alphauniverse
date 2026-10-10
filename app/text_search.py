from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from .config import DIM, GEMMA, GEMMA_DIM, PAIRS, RGB_COLUMN, artifact, device
from .dataset import dataset

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BATCH = 1024


class TextQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1, max_length=500)]
    matches: Annotated[int, Field(ge=1, le=128)] = 32


@cache
def text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(GEMMA, device=device())


def embed_queries(texts: list[str]) -> np.ndarray:
    return text_model().encode(texts, prompt_name="SearchQuery")


def generate_pairs() -> None:
    from .parametric_umap import _stream

    data = dataset()
    sums = np.zeros((len(data), DIM), dtype=np.float64)
    held = np.zeros(len(data), dtype=np.int64)
    for galaxies, offsets, values in tqdm(_stream(), desc="mean"):
        sums[galaxies] += np.add.reduceat(values, offsets[:-1])
        held[galaxies] += np.diff(offsets)
    aion = (sums / held[:, None]).astype(np.float32)
    images = data.select_columns([RGB_COLUMN]).cast_column(RGB_COLUMN, Image())
    gemma = np.concatenate(
        [
            text_model().encode(batch[RGB_COLUMN], prompt_name="Document")
            for batch in images.iter(batch_size=BATCH)
        ]
    )

    pq.write_table(
        pa.table(
            {
                "galaxy": np.arange(len(data), dtype=np.int32),
                "aion": pa.FixedSizeListArray.from_arrays(aion.ravel(), DIM),
                "gemma": pa.FixedSizeListArray.from_arrays(gemma.ravel(), GEMMA_DIM),
            },
            schema=PAIRS,
        ),
        artifact("pairs"),
        compression="zstd",
    )


@cache
def aion_gemma_space() -> np.ndarray:
    return np.load(artifact("aion_gemma_space"))


def text_search(query: TextQuery) -> tuple[np.ndarray, np.ndarray]:
    scores = aion_gemma_space() @ embed_queries([query.text])[0]
    count = min(query.matches, len(scores))
    top = np.argpartition(-scores, count - 1)[:count]
    order = top[np.argsort(-scores[top], kind="stable")]
    return order.astype(np.int32), scores[order]
