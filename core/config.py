from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from aion.fourm.modality_info import MODALITY_INFO

DATASET_AUTHOR = "hanbangwu"

DATASET_NAME = "alphauniverse-cosmos"

DATASET_REVISION = "e250e43c35e63523ee3940c9543302c29ca56437"

DATASET_LS_SUFFIX = "-mmu_legacysurvey_dr10_south_21"

DATASET_HSC_SUFFIX = "-mmu_hsc_pdr3_dud_22.5"

DATASET_DESI_SUFFIX = "-mmu_desi_edr_sv3"

DATASET_SDSS_SUFFIX = "-mmu_sdss_sdss"

DATASET_GZ10_SUFFIX = "-mmu_gz10"

DATASET_PROVABGS_SUFFIX = "-mmu_desi_provabgs"

AION_SPECTRUM_ORIGIN = 3500.0

AION_SPECTRUM_PIXEL = 0.8

AION_SPECTRUM_GRID = AION_SPECTRUM_ORIGIN + AION_SPECTRUM_PIXEL * np.arange(8704)

AION_SPECTRUM_TOKEN_WIDTH = 32 * AION_SPECTRUM_PIXEL

SPECTRUM_SMOOTHING_SIGMA = 2

SPECTRUM_SURVEY = "desi"

SpectrumSurvey = Literal["desi", "sdss"]

ImageSurvey = Literal["ls", "hsc"]

TOKEN_SURVEYS: dict[str, str] = {
    "ls": f"image{DATASET_LS_SUFFIX}",
    "hsc": f"image{DATASET_HSC_SUFFIX}",
    "desi": f"spectrum{DATASET_DESI_SUFFIX}",
    "sdss": f"spectrum{DATASET_SDSS_SUFFIX}",
}

FLAG_SURVEYS: dict[str, str] = {
    "gz10": f"gz10_label{DATASET_GZ10_SUFFIX}",
    "provabgs": f"LOG_MSTAR{DATASET_PROVABGS_SUFFIX}",
}

LS_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")

HSC_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")

LS_TABLE_COLUMNS = tuple(
    f"{name}{DATASET_LS_SUFFIX}"
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
)

HSC_TABLE_COLUMNS = tuple(
    f"{name}{DATASET_HSC_SUFFIX}"
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
)

REDSHIFT = "redshift"

REDSHIFT_LIMIT = 6.0

REDSHIFT_COLUMNS: dict[SpectrumSurvey, tuple[str, str, str]] = {
    "desi": (
        f"Z{DATASET_DESI_SUFFIX}",
        f"ZERR{DATASET_DESI_SUFFIX}",
        f"ZWARN{DATASET_DESI_SUFFIX}",
    ),
    "sdss": (
        f"Z{DATASET_SDSS_SUFFIX}",
        f"Z_ERR{DATASET_SDSS_SUFFIX}",
        f"ZWARNING{DATASET_SDSS_SUFFIX}",
    ),
}

WEMM = "tencent/WeMM-Embedding-9B"

WEMM_DIM = 1024

QUERY_PREFIX = "Search galaxy spectra, images and catalogue measurements for: {}"

TextModality = Literal["image", "spectrum"]

TEXT_SURVEYS: dict[TextModality, str] = {
    "image": "ls",
    "spectrum": SPECTRUM_SURVEY,
}

TOP_CODES = 64

KEPT = 16

PROBABILITY_FLOOR = float(np.finfo(np.float32).tiny)

SPECTRUM_TOKEN_RANK = 256

IMAGE_MODES = {f"{survey}_image": survey for survey in get_args(ImageSurvey)}

SPECTRUM_MODES = {f"{survey}_spectrum": survey for survey in get_args(SpectrumSurvey)}

TokenMode = Literal[*IMAGE_MODES, *SPECTRUM_MODES]

TABLE_MODES = {REDSHIFT: (REDSHIFT, 1, MODALITY_INFO["tok_z"]["vocab_size"])} | {
    f"{survey}_table": (
        f"{survey}_table_values",
        len(columns),
        MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
    )
    for survey, columns in (("ls", LS_TABLE_COLUMNS), ("hsc", HSC_TABLE_COLUMNS))
}

MODE_SURVEYS = (
    IMAGE_MODES
    | SPECTRUM_MODES
    | {f"{survey}_table": survey for survey in get_args(ImageSurvey)}
    | {REDSHIFT: REDSHIFT}
)

MODES = (*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES)

TOKEN_GRAMS = {
    mode: (mode, f"{mode}_anywhere") if mode in get_args(TokenMode) else (mode,)
    for mode in MODES
}

GRAM_NAMES = tuple(name for names in TOKEN_GRAMS.values() for name in names)

MAPS: dict[str, frozenset[str]] = {
    "same_place": frozenset(),
    "any_position": frozenset(get_args(TokenMode)),
}

SEARCH_RANK = 256
