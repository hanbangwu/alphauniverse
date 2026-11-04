from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import cache
from itertools import batched, product
from threading import Lock
from typing import TYPE_CHECKING, Annotated, NamedTuple

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import Image
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer
from tqdm import tqdm

from .config import (
    DIM,
    GEMMA,
    GEMMA_DIM,
    N_SPECTRUM_TOKENS,
    PAIRS,
    PLOT_INCHES,
    QUERY_PREFIX,
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
LOADING = Lock()
FRAMES = {
    "everything": "{}",
    "images": "an image of {}",
    "spectra": "the spectrum of {}",
    "catalogues": "the photometry of {}",
}
ADJECTIVES = (
    "",
    "bright",
    "faint",
    "red",
    "blue",
    "massive",
    "small",
    "large",
    "nearby",
    "distant",
    "dusty",
    "star-forming",
    "quiescent",
    "compact",
)
SUBJECTS = (
    "galaxy",
    "spiral galaxy",
    "elliptical galaxy",
    "disk galaxy",
    "irregular galaxy",
    "dwarf galaxy",
    "lenticular galaxy",
    "barred spiral galaxy",
    "pair of merging galaxies",
    "edge-on galaxy",
    "face-on spiral galaxy",
    "starburst galaxy",
    "quasar host galaxy",
    "cluster galaxy",
    "low surface brightness galaxy",
)
SUFFIXES = (
    "",
    " with spiral arms",
    " with a bright bulge",
    " with a central bar",
    " with tidal tails",
    " with strong emission lines",
    " with an old stellar population",
    " at high redshift",
    " at low redshift",
    " with a dust lane",
    " with a close companion",
    " with an active nucleus",
)


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
        GEMMA,
        device=device(),
        trust_remote_code=True,
        truncate_dim=GEMMA_DIM,
        model_kwargs=kernels,
    )
    trained = Tokenizer.from_file(hf_hub_download(GEMMA, "tokenizer.json"))
    model[0].processor.tokenizer.backend_tokenizer.normalizer = trained.normalizer
    return model


def embed_queries(texts: list[str]) -> np.ndarray:
    with LOADING:
        model = text_model()
    return model.encode_query(
        [QUERY_PREFIX.format(text) for text in texts], normalize_embeddings=True
    )


def generic_queries() -> list[str]:
    phrases = (
        f"{adjective} {subject}".strip()
        for adjective, subject in product(ADJECTIVES, SUBJECTS)
    )
    nouns = [("an " if phrase[0] in "aeiou" else "a ") + phrase for phrase in phrases]
    return [
        frame.format(noun + suffix)
        for frame, noun, suffix in product(FRAMES.values(), nouns, SUFFIXES)
    ]


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
                text_model().encode_document(batch, normalize_embeddings=True)
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
