from collections.abc import Iterator
from io import BytesIO
from itertools import batched

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from aion.modalities import DESISpectrum
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from app.config import (
    AION_EMBEDDING_DIM,
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    PAIRS,
    PLOT_INCHES,
    SEED,
    SPECTRUM_SURVEY,
    TEXT_SURVEYS,
    TOKEN_SURVEYS,
    WEMM_DIM,
    TextModality,
    artifact,
    device,
)
from app.dataset.dataset import dataset, image, samples, smoothed
from app.encoding.store import stream
from app.text_search.text_search import BATCH, text_model

VALIDATION = 0.3


def pairs(modality: TextModality) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    table = pq.read_table(
        artifact(f"pairs_{modality}"), columns=["galaxy", "aion", "wemm"]
    )
    galaxy = np.asarray(table.column("galaxy"))
    aion, wemm = (
        torch.as_tensor(
            np.asarray(table.column(name).combine_chunks().flatten()).reshape(
                len(galaxy), -1
            ),
            device=device(),
        )
        for name in ("aion", "wemm")
    )
    return galaxy, aion, wemm


def held_out(galaxy: np.ndarray) -> torch.Tensor:
    _, validation = train_test_split(
        np.arange(len(dataset())), test_size=VALIDATION, random_state=SEED
    )
    return torch.as_tensor(np.isin(galaxy, validation), device=device())


def spectrum_plot(spectrum: dict[str, np.ndarray]) -> PIL.Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=PLOT_INCHES)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(spectrum["wavelength"], smoothed(spectrum["flux"]), color="gray")
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
    count = len(dataset())
    means = np.zeros((count, AION_EMBEDDING_DIM), dtype=np.float32)
    held = np.zeros(count, dtype=bool)
    for galaxies, offsets, values in tqdm(stream((survey,)), desc=survey):
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
