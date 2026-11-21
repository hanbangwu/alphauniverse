from collections.abc import Iterator
from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from aion.codecs.quantizers import FiniteScalarQuantizer, LucidrainsLFQ
from aion.fourm.modality_info import MODALITY_INFO
from pyarrow.fs import LocalFileSystem

from app.config import (
    AION_EMBEDDING_DIM,
    ARTIFACTS,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    REDSHIFT_TABLE_VALUE,
    STORE_COLUMNS,
    artifact,
)

N_LS_TABLE_VALUES = len(LS_TABLE_COLUMNS)
N_HSC_TABLE_VALUES = len(HSC_TABLE_COLUMNS)
FIRST_LS_TABLE_VALUE = REDSHIFT_TABLE_VALUE + 1
FIRST_HSC_TABLE_VALUE = FIRST_LS_TABLE_VALUE + N_LS_TABLE_VALUES
CHUNK = 128
IMAGE_CODE_LEVELS = [7, 5, 5, 5, 5]
SPECTRUM_CODE_DIM = 10


@cache
def source(role: str) -> ds.Dataset:
    return ds.dataset(
        artifact(role),
        format=ARTIFACTS[role],
        filesystem=LocalFileSystem(use_mmap=True),
    )


@cache
def tokens() -> pa.Table:
    return source("tokens").to_table()


@cache
def observed(column: str) -> np.ndarray:
    return pc.is_valid(tokens().column(column)).to_numpy()


@cache
def image_lattice() -> np.ndarray:
    codes = torch.arange(MODALITY_INFO["tok_image"]["vocab_size"])
    return FiniteScalarQuantizer(levels=IMAGE_CODE_LEVELS).decode(codes).numpy()


@cache
def spectrum_lattice() -> np.ndarray:
    codes = torch.arange(MODALITY_INFO["tok_spectrum_desi"]["vocab_size"])
    quantizer = LucidrainsLFQ(dim=SPECTRUM_CODE_DIM, codebook_size=len(codes))
    return quantizer.decode(codes).numpy()


def false_colors(latents: np.ndarray) -> np.ndarray:
    centered = latents - latents.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    components = centered @ directions[:3].T
    low, high = components.min(axis=0), components.max(axis=0)
    scaled = np.divide(
        components - low,
        high - low,
        out=np.zeros_like(components),
        where=high > low,
    )
    return np.round(scaled * 255).astype(np.uint8)


def stream(
    surveys: tuple[str, ...] = STORE_COLUMNS,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for survey in surveys:
        for stored in source("encoded").to_batches(
            columns=["galaxy", survey], batch_size=CHUNK
        ):
            batch = stored.filter(stored.column(survey).is_valid())
            cells = batch.column(survey)
            offsets = np.asarray(cells.offsets, dtype=np.intp)
            yield (
                np.asarray(batch.column("galaxy")),
                offsets - offsets[0],
                np.asarray(cells.flatten().flatten(), dtype=np.float32).reshape(
                    -1, AION_EMBEDDING_DIM
                ),
            )
