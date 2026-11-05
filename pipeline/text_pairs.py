from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from itertools import batched

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
from aion.modalities import DESISpectrum
from tqdm import tqdm

from core.artifacts import artifact
from core.config import (
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    SPECTRUM_SURVEY,
    TEXT_SURVEYS,
    TOKEN_SURVEYS,
    WEMM_DIM,
)
from core.dataset import dataset, image, samples, smoothed
from core.text_model import BATCH, text_model

from .config import AION_EMBEDDING_DIM, PLOT_INCHES
from .schemas import PAIRS


def spectrum_plot(spectrum: dict[str, np.ndarray]) -> PIL.Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=PLOT_INCHES)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(spectrum["wavelength"], smoothed(spectrum["flux"]), color="grey")
    axes.set_xlim(
        AION_SPECTRUM_ORIGIN,
        AION_SPECTRUM_ORIGIN
        + (DESISpectrum.num_tokens - 1) * AION_SPECTRUM_TOKEN_WIDTH,
    )
    canvas.draw()
    return PIL.Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def photos(galaxy: np.ndarray) -> Iterator[list[PIL.Image.Image]]:
    for batch in batched(galaxy.tolist(), BATCH):
        yield [PIL.Image.open(BytesIO(image(index, "ls"))) for index in batch]


def plots(galaxy: np.ndarray) -> Iterator[list[PIL.Image.Image]]:
    cells = dataset().data.column(TOKEN_SURVEYS[SPECTRUM_SURVEY])
    for batch in batched(galaxy.tolist(), BATCH):
        yield [spectrum_plot(samples(cells[index])) for index in batch]


DOCUMENTS = {"image": photos, "spectrum": plots}


def mean_embeddings(survey: str) -> tuple[np.ndarray, np.ndarray]:
    from .parametric_umap import _stream

    count = len(dataset())
    means = np.zeros((count, AION_EMBEDDING_DIM), dtype=np.float32)
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
        wemm = np.concatenate(
            [
                text_model().encode_document(batch, normalize_embeddings=True)
                for batch in tqdm(DOCUMENTS[modality](galaxy), desc=modality)
            ]
        )
        pq.write_table(
            pa.table(
                {
                    "galaxy": galaxy,
                    "aion": pa.FixedSizeListArray.from_arrays(
                        aion.ravel(), AION_EMBEDDING_DIM
                    ),
                    "wemm": pa.FixedSizeListArray.from_arrays(wemm.ravel(), WEMM_DIM),
                },
                schema=PAIRS,
            ),
            artifact(f"pairs_{modality}"),
            compression="zstd",
        )
