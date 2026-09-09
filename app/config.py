from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

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

BUILD_DIR = (
    Path(
        os.environ.get(
            "ALPHAUNIVERSE_CACHE", Path(__file__).resolve().parent.parent / ".cache"
        )
    )
    / DATASET_AUTHOR
    / DATASET_NAME
    / DATASET_REVISION
)

CROP_PX = 96

DIM = 768

GRID = 24
N_PATCHES = GRID**2

GalaxyIndex = Annotated[int, Field(ge=0, lt=17369)]

ANCHOR = "ls"
LS = "-mmu_legacysurvey_dr10_south_21"
HSC = "-mmu_hsc_pdr3_dud_22.5"
DESI = "-mmu_desi_edr_sv3"
SDSS = "-mmu_sdss_sdss"
GZ10 = "-mmu_gz10"
PROVABGS = "-mmu_desi_provabgs"


TOKEN_SURVEYS: dict[str, str] = {
    ANCHOR: f"image{LS}",
    "hsc": f"image{HSC}",
    "desi": f"spectrum{DESI}",
    "sdss": f"spectrum{SDSS}",
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

ARTIFACTS: dict[str, str] = {
    "encoded": "parquet",
    "encoded_index": "faiss",
    "codebook": "parquet",
    "tokens": "parquet",
    "mean_points": "parquet",
    "full_points": "parquet",
    "parametric_umap": "pt",
}


def artifact(role: str) -> Path:
    return BUILD_DIR / f"{role}.{ARTIFACTS[role]}"


WANDB_ENTITY = "aistrophysics"
WANDB_PROJECT = "alphaUniverse"
WANDB_MODE = os.environ.get(
    "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled"
)

SEED = 42
