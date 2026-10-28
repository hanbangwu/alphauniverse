from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from itertools import batched
from typing import TYPE_CHECKING, Annotated, NamedTuple

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from .config import (
    DIM,
    GEMMA,
    GEMMA_DIM,
    N_SPECTRUM_TOKENS,
    PAIRS,
    PLOT_INCHES,
    RGB_COLUMN,
    SPECTRUM_ORIGIN,
    SPECTRUM_SURVEY,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_WIDTH,
    TEXT_SURVEYS,
    artifact,
    device,
)
from .dataset import dataset, samples, smoothed

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BATCH = 1024


class TextQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1, max_length=500)]
    matches: Annotated[int, Field(ge=1, le=128)] = 32


class TextSpace(NamedTuple):
    vectors: np.ndarray
    galaxy: np.ndarray
    modality: np.ndarray


@cache
def text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(GEMMA, device=device())


def embed_queries(texts: list[str]) -> np.ndarray:
    return text_model().encode(texts, prompt_name="SearchQuery")


def spectrum_plot(spectrum: dict[str, np.ndarray]) -> PIL.Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=PLOT_INCHES)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(spectrum["wavelength"], smoothed(spectrum["flux"]), color="grey")
    axes.set_xlim(
        SPECTRUM_ORIGIN, SPECTRUM_ORIGIN + N_SPECTRUM_TOKENS * SPECTRUM_TOKEN_WIDTH
    )
    canvas.draw()
    return PIL.Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def photos(galaxy: np.ndarray) -> Iterator[list[PIL.Image.Image]]:
    images = (
        dataset()
        .select_columns([RGB_COLUMN])
        .select(galaxy)
        .cast_column(RGB_COLUMN, Image())
    )
    for batch in images.iter(batch_size=BATCH):
        yield batch[RGB_COLUMN]


def plots(galaxy: np.ndarray) -> Iterator[list[PIL.Image.Image]]:
    cells = dataset().data.column(SPECTRUM_SURVEYS[SPECTRUM_SURVEY])
    for batch in batched(galaxy.tolist(), BATCH):
        yield [spectrum_plot(samples(cells[index])) for index in batch]


DOCUMENTS = {"image": photos, "spectrum": plots}


def mean_embeddings(survey: str) -> tuple[np.ndarray, np.ndarray]:
    from .parametric_umap import _stream

    count = len(dataset())
    means = np.zeros((count, DIM), dtype=np.float32)
    held = np.zeros(count, dtype=bool)
    for galaxies, offsets, values in tqdm(_stream((survey,)), desc=survey):
        sizes = np.diff(offsets)[:, None]
        means[galaxies] = np.add.reduceat(values, offsets[:-1]) / sizes
        held[galaxies] = True
    galaxy = np.flatnonzero(held).astype(np.int32)
    return galaxy, means[galaxy]


def generate_pairs() -> None:
    for modality, survey in TEXT_SURVEYS.items():
        galaxy, aion = mean_embeddings(survey)
        gemma = np.concatenate(
            [
                text_model().encode(batch, prompt_name="Document")
                for batch in tqdm(DOCUMENTS[modality](galaxy), desc=modality)
            ]
        )
        pq.write_table(
            pa.table(
                {
                    "galaxy": galaxy,
                    "aion": pa.FixedSizeListArray.from_arrays(aion.ravel(), DIM),
                    "gemma": pa.FixedSizeListArray.from_arrays(
                        gemma.ravel(), GEMMA_DIM
                    ),
                },
                schema=PAIRS,
            ),
            artifact(f"pairs_{modality}"),
            compression="zstd",
        )


@cache
def aion_gemma_space() -> TextSpace:
    with np.load(artifact("aion_gemma_space")) as stored:
        return TextSpace(**stored)


def text_search(query: TextQuery) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    space = aion_gemma_space()
    scores = space.vectors @ embed_queries([query.text])[0]
    count = min(query.matches, len(scores))
    top = np.argpartition(-scores, count - 1)[:count]
    order = top[np.argsort(-scores[top], kind="stable")]
    return space.galaxy[order], space.modality[order], scores[order]
