from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pyarrow as pa
from pydantic import Field

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
DATASET_ID = f"{DATASET_AUTHOR}/{DATASET_NAME}"
DATASET_REVISION = "e250e43c35e63523ee3940c9543302c29ca56437"

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache"

CROP_PIXELS = 96

DIM = 768

GRID = 24
N_PATCHES = GRID**2

SPECTRUM_ORIGIN = 3500.0
SPECTRUM_TOKEN_WIDTH = 32 * 0.8
N_SPANS = 8704 // 32

GALAXIES = 17369
GalaxyIndex = Annotated[int, Field(ge=0, lt=GALAXIES)]

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

N_MORPHOLOGIES = 10

NLIST = 16384
NPROBE = 64
PROBE = 2048
TRAIN_GALAXIES = 2048
BATCH = 256
MIN_TRAIN_PER_CENTROID = 39

ARTIFACTS: dict[str, str] = {
    "encoded": "parquet",
    "encoded_index": "faiss",
    "cutouts": "parquet",
    "spectra": "parquet",
    "codebook": "parquet",
    "tokens": "parquet",
    "mean_points": "parquet",
    "full_points": "parquet",
    "parametric_umap": "pt",
}


CUTOUTS = pa.schema(
    [pa.field("galaxy", pa.int32()), pa.field("png", pa.large_binary())]
)

SPECTRUM = pa.struct(
    [
        pa.field("wavelength", pa.list_(pa.float32())),
        pa.field("flux", pa.list_(pa.float32())),
    ]
)
SPECTRA = pa.schema(
    [pa.field("galaxy", pa.int32())]
    + [pa.field(survey, SPECTRUM) for survey in SPECTRUM_SURVEYS]
)

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


POINTS = pa.schema(
    [
        pa.field("galaxy", pa.int32(), nullable=False),
        pa.field("x", pa.float32(), nullable=False),
        pa.field("y", pa.float32(), nullable=False),
        pa.field("category", pa.uint8(), nullable=True),
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


WANDB_ENTITY = "aistrophysics"
WANDB_PROJECT = "alphaUniverse"
WANDB_MODE = os.environ.get(
    "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled"
)

SEED = 42
