from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import AfterValidator, Field

if TYPE_CHECKING:
    import torch


@cache
def device() -> torch.device:
    import torch

    return torch.accelerator.current_accelerator(check_available=True) or torch.device(
        "cpu"
    )


DATASET_AUTHOR = "hanbangwu"
DATASET_NAME = "alphauniverse-cosmos"
DATASET_REVISION = "e250e43c35e63523ee3940c9543302c29ca56437"


DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache"

CROP_PIXELS = 96

DIM = 768

GRID = 24
N_PATCHES = GRID**2

SPECTRUM_ORIGIN = 3500.0
SPECTRUM_SMOOTHING_SIGMA = 2
SPECTRUM_SURVEY: SpectrumSurvey = "desi"
SPECTRUM_TOKEN_WIDTH = 32 * 0.8
N_SPANS = 8704 // 32

ANCHOR = "ls"
LS = "-mmu_legacysurvey_dr10_south_21"
HSC = "-mmu_hsc_pdr3_dud_22.5"
DESI = "-mmu_desi_edr_sv3"
SDSS = "-mmu_sdss_sdss"
GZ10 = "-mmu_gz10"
PROVABGS = "-mmu_desi_provabgs"


SpectrumSurvey = Literal["desi", "sdss"]
SPECTRUM_SURVEYS: dict[SpectrumSurvey, str] = {
    "desi": f"spectrum{DESI}",
    "sdss": f"spectrum{SDSS}",
}
TOKEN_SURVEYS: dict[str, str] = {
    ANCHOR: f"image{LS}",
    "hsc": f"image{HSC}",
    **SPECTRUM_SURVEYS,
}
FLAG_SURVEYS: dict[str, str] = {
    "gz10": f"gz10_label{GZ10}",
    "provabgs": f"LOG_MSTAR{PROVABGS}",
}
RGB_COLUMN = f"rgb{LS}"

SCALAR_SURVEYS: dict[str, tuple[str, ...]] = {
    ANCHOR: tuple(
        f"{name}{LS}"
        for name in (
            "EBV",
            "FLUX_G",
            "FLUX_R",
            "FLUX_I",
            "FLUX_Z",
            "FLUX_W1",
            "FLUX_W2",
            "FLUX_W3",
            "FLUX_W4",
            "SHAPE_R",
            "SHAPE_E1",
            "SHAPE_E2",
        )
    ),
    "hsc": tuple(
        f"{name}{HSC}"
        for name in (
            "a_g",
            "a_r",
            "a_i",
            "a_z",
            "a_y",
            "g_cmodel_mag",
            "r_cmodel_mag",
            "i_cmodel_mag",
            "z_cmodel_mag",
            "y_cmodel_mag",
            "i_sdssshape_shape11",
            "i_sdssshape_shape22",
            "i_sdssshape_shape12",
        )
    ),
}
SCALAR_COLUMNS = tuple(
    column for columns in SCALAR_SURVEYS.values() for column in columns
)
N_SCALARS = len(SCALAR_COLUMNS)

Catalogue = Literal["ls", "hsc", "desi", "sdss", "gz10", "provabgs"]
CATALOGUES: dict[str, Catalogue] = {
    LS: ANCHOR,
    HSC: "hsc",
    DESI: "desi",
    SDSS: "sdss",
    GZ10: "gz10",
    PROVABGS: "provabgs",
}

N_MORPHOLOGIES = 10

GEMMA = "google/embeddinggemma-2"
GEMMA_DIM = 768

ARTIFACTS: dict[str, str] = {
    "encoded": "arrow",
    "search_index": "faiss",
    "codebook": "parquet",
    "tokens": "arrow",
    "mean_points": "parquet",
    "full_points": "parquet",
    "parametric_umap": "pt",
    "pairs": "parquet",
    "alignment": "pt",
    "aion_gemma_space": "npy",
}
Projection = Literal["mean", "full"]
Download = Literal["encoded", "codebook", "tokens"]


STORES = ("encoded", "codebook", "tokens")


def store_schema(role: str) -> pa.Schema:
    cell = (
        pa.list_(pa.uint32())
        if role == "tokens"
        else pa.list_(pa.list_(pa.float16(), DIM))
    )
    return pa.schema(
        [pa.field("galaxy", pa.int32())]
        + [pa.field(survey, cell) for survey in TOKEN_SURVEYS]
        + [pa.field(survey, pa.bool_()) for survey in FLAG_SURVEYS]
    )


def store_writer(role: str) -> pq.ParquetWriter | pa.ipc.RecordBatchFileWriter:
    if ARTIFACTS[role] == "arrow":
        return pa.ipc.new_file(artifact(role), store_schema(role))
    return pq.ParquetWriter(artifact(role), store_schema(role), compression="zstd")


PAIRS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("aion", pa.list_(pa.float32(), DIM)),
        pa.field("gemma", pa.list_(pa.float32(), GEMMA_DIM)),
    ]
)


POINTS = pa.schema(
    [
        pa.field("galaxy", pa.int32(), nullable=False),
        pa.field("x", pa.float32(), nullable=False),
        pa.field("y", pa.float32(), nullable=False),
        pa.field("category", pa.uint8()),
    ]
)


def points(galaxy: np.ndarray, coordinates: np.ndarray, category: pa.Array) -> pa.Table:
    return pa.table(
        {
            "galaxy": galaxy,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "category": category,
        },
        schema=POINTS,
    )


def build_dir() -> Path:
    return (
        Path(os.environ.get("ALPHAUNIVERSE_CACHE", DEFAULT_CACHE))
        / DATASET_AUTHOR
        / DATASET_NAME
        / DATASET_REVISION
    )


def artifact(role: str) -> Path:
    return build_dir() / f"{role}.{ARTIFACTS[role]}"


@cache
def galaxy_count() -> int:
    return pq.read_metadata(artifact("mean_points")).num_rows


@cache
def labels() -> list[int]:
    category = pq.read_table(artifact("mean_points"), columns=["category"])["category"]
    return np.bincount(
        np.asarray(category.fill_null(N_MORPHOLOGIES)), minlength=N_MORPHOLOGIES + 1
    ).tolist()


def stored_galaxy(galaxy: int) -> int:
    if galaxy >= galaxy_count():
        raise ValueError(f"should be less than {galaxy_count()}, the galaxy count")
    return galaxy


GalaxyIndex = Annotated[int, Field(ge=0), AfterValidator(stored_galaxy)]


WANDB_ENTITY = "aistrophysics"
WANDB_PROJECT = "alphaUniverse"
WANDB_MODE = os.environ.get(
    "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled"
)

SEED = 42
