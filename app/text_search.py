from __future__ import annotations

import math
from collections.abc import Iterator
from functools import cache
from itertools import batched, product
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import Image
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from .config import (
    DIM,
    GEMMA,
    GEMMA_DIM,
    MODES,
    N_SPECTRUM_TOKENS,
    PAIRS,
    PLOT_INCHES,
    RGB_COLUMN,
    SPECTRUM_ORIGIN,
    SPECTRUM_SURVEY,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_WIDTH,
    TEXT_SURVEYS,
    VIEW_COMBINATIONS,
    VIEW_TYPES,
    ModeName,
    artifact,
    device,
)
from .dataset import dataset, samples, smoothed

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BATCH = 1024
FRAMES = ("{}", "an image of {}", "the spectrum of {}", "the photometry of {}")
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

ShareName = Literal[*MODES, "combined"]


class TextQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1, max_length=500)]
    matches: Annotated[int, Field(ge=1, le=128)] = 32
    modes: Annotated[frozenset[ModeName], Field(min_length=1)] = frozenset(MODES)


class TextSpace(NamedTuple):
    vectors: np.ndarray
    galaxy: np.ndarray
    modality: np.ndarray


class TextIndex(NamedTuple):
    points: torch.Tensor
    lengths: torch.Tensor
    components: torch.Tensor
    centre: torch.Tensor
    query_mean: torch.Tensor
    kappa: float


class TextResults(NamedTuple):
    galaxies: torch.Tensor
    scores: torch.Tensor
    probabilities: torch.Tensor
    shares: dict[ShareName, torch.Tensor]
    breadth: float


@cache
def text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(GEMMA, device=device())


def embed_queries(texts: list[str]) -> np.ndarray:
    return text_model().encode(texts, prompt_name="SearchQuery")


def generic_queries() -> list[str]:
    phrases = (
        f"{adjective} {subject}".strip()
        for adjective, subject in product(ADJECTIVES, SUBJECTS)
    )
    nouns = [("an " if phrase[0] in "aeiou" else "a ") + phrase for phrase in phrases]
    return [
        frame.format(noun + suffix)
        for frame, noun, suffix in product(FRAMES, nouns, SUFFIXES)
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


def point_lengths(
    points: torch.Tensor, components: torch.Tensor, centre: torch.Tensor
) -> torch.Tensor:
    squared = points.square().sum(dim=-1) + centre @ centre
    return (squared + 2 * points @ (components @ centre)).sqrt()


@cache
def text_index() -> TextIndex:
    with np.load(artifact("text_bases")) as stored:
        components = torch.from_numpy(stored["components"])
        centre = torch.from_numpy(stored["centre"])
        kappa = float(stored["kappa"])
    clouds = np.load(artifact("text_clouds"), mmap_mode="r")
    points = torch.empty(clouds.shape)
    lengths = torch.empty(clouds.shape[:-1])
    for start in range(0, clouds.shape[1], BATCH):
        rows = slice(start, start + BATCH)
        points[:, rows] = torch.from_numpy(clouds[:, rows].astype(np.float32))
        for kind in range(len(VIEW_TYPES)):
            lengths[kind, rows] = point_lengths(
                points[kind, rows], components[kind], centre[kind]
            )
    query_mean = torch.from_numpy(np.load(artifact("text_query_mean")))
    return TextIndex(points, lengths, components, centre, query_mean, kappa)


def view_modes(kind: str, name: str) -> tuple[str, ...]:
    return VIEW_COMBINATIONS[name] if kind == "combination" else (name,)


def cloud_scores(
    index: TextIndex, direction: torch.Tensor, kinds: list[int], kappa: float
) -> tuple[torch.Tensor, torch.Tensor]:
    along = index.components @ direction
    offsets = index.centre @ direction
    logits = torch.stack(
        [
            kappa
            * (index.points[kind] @ along[kind] + offsets[kind])
            / index.lengths[kind]
            for kind in kinds
        ],
        dim=1,
    ).flatten(1)
    return logits.logsumexp(dim=1) - math.log(logits.shape[1]), logits


def shares(weights: torch.Tensor, kinds: list[int]) -> dict[ShareName, torch.Tensor]:
    owners = [
        "combined" if VIEW_TYPES[kind][0] == "combination" else VIEW_TYPES[kind][1]
        for kind in kinds
    ]
    return {
        name: weights[
            :, [column for column, owner in enumerate(owners) if owner == name]
        ].sum(dim=1)
        for name in (*MODES, "combined")
    }


def breadth(scores: torch.Tensor) -> float:
    logged = scores.log_softmax(dim=0)
    return math.exp(-(logged.exp() * logged).sum().item()) / len(scores)


def text_search(query: TextQuery) -> TextResults:
    index = text_index()
    embedded = torch.from_numpy(embed_queries([query.text])[0])
    direction = F.normalize(embedded - index.query_mean, dim=0)
    kinds = [
        kind
        for kind, view in enumerate(VIEW_TYPES)
        if set(view_modes(*view)) <= query.modes
    ]
    scores, logits = cloud_scores(index, direction, kinds, index.kappa)
    top = scores.topk(min(query.matches, len(scores))).indices
    weights = logits[top].softmax(dim=1).unflatten(1, (len(kinds), -1)).sum(dim=2)
    return TextResults(
        top,
        scores[top],
        scores.softmax(dim=0)[top],
        shares(weights, kinds),
        breadth(scores),
    )
