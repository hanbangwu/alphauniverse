from __future__ import annotations

import os

import numpy as np

from core.config import (
    IMAGE_MODES,
    MODES,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_MODES,
    TOKEN_SURVEYS,
)

LS_ZERO_POINT = 22.5

HSC_ZERO_POINT = 27.0

AION_EMBEDDING_DIM = 768

STORE_COLUMNS = (*TOKEN_SURVEYS, REDSHIFT)

PLOT_INCHES = (16.8, 3.84)

STORES = ("encoded", "tokens")

VIEW_KINDS = ("observed", "aion_prediction", "sample")

VIEW_COMBINATIONS = {
    "images": tuple(IMAGE_MODES),
    "spectra": tuple(SPECTRUM_MODES),
    "catalogues": tuple(TABLE_MODES),
    "image_and_spectrum": ("ls_image", "desi_spectrum"),
    "everything": MODES,
}

VIEW_TYPES = (
    *((kind, mode) for mode in MODES for kind in VIEW_KINDS),
    *(("combination", name) for name in VIEW_COMBINATIONS),
)

TEXT_RANK = 128

TEXT_CONDITION = 256

VIEWS_PER_PART = 4_000

PARTS_PER_ROUND = 10

PROTOTYPE_HELD_OUT = np.concatenate(
    [
        [110, 301, 319, 830, 1003, 1365, 1413, 1492, 1561, 1653, 1655, 1656, 1790],
        [1843, 1918, 1988, 1989, 1991, 2006, 2026, 2027, 2092, 2173, 2219, 2247, 2414],
        [2481, 2725, 2726, 2863, 2918, 3095, 3166, 3221, 3286, 3333, 3923, 3955, 3962],
        [3965, 4093, 4102, 4459, 4670, 4891, 4926, 5026, 5091, 5179, 5238, 5245, 5287],
        [5371, 5467, 5643, 5670, 5678, 5710, 5932, 6313, 6319, 6694, 6705, 7134, 7139],
        [7184, 7190, 7493, 7520, 7650, 7666, 7810, 7896, 7938, 7983, 8081, 8087, 8123],
        [8226, 8384, 8414, 8494, 8517, 8680, 9305, 9452, 9679, 9701, 9938, 10004],
        [10097, 10104, 10469, 10474, 10602, 10716, 10721, 10745, 10942, 11218, 11581],
        [12137, 12230, 12576, 12628, 12641, 12660, 12689, 12846, 12916, 12970, 13105],
        [13190, 13455, 13471, 13540, 13545, 13592, 13989, 14103, 14113, 14268, 14308],
        [14381, 14383, 14631, 14680, 14733, 14908, 15169, 15304, 15307, 15598, 15614],
        [15768, 16021, 16075, 16236, 16304, 16464, 16928, 17239, 17243],
    ]
)

TOKEN_EMBEDDING_RANK = 1024

PROVABGS_PROPERTIES = ("mass", "sfr", "ssfr")

LS_PROPERTIES = ("radius", "r_magnitude", "axis_ratio")

SPECTRAL_PROPERTIES = ("halpha_ew", "dn4000")

PROBED = (*PROVABGS_PROPERTIES, *LS_PROPERTIES, *SPECTRAL_PROPERTIES)

WANDB_ENTITY = "aistrophysics"

WANDB_PROJECT = "alphaUniverse"

WANDB_MODE = os.environ.get(
    "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled"
)

SEED = 42
