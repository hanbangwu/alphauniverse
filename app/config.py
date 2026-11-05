from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage
from annotated_types import Predicate
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
DATASET_REVISION = "e250e43c35e63523ee3940c9543302c29ca56437"
DATASET_LS_SUFFIX = "-mmu_legacysurvey_dr10_south_21"
DATASET_HSC_SUFFIX = "-mmu_hsc_pdr3_dud_22.5"
DATASET_DESI_SUFFIX = "-mmu_desi_edr_sv3"
DATASET_SDSS_SUFFIX = "-mmu_sdss_sdss"
DATASET_GZ10_SUFFIX = "-mmu_gz10"
DATASET_PROVABGS_SUFFIX = "-mmu_desi_provabgs"
LS_ZERO_POINT = 22.5
HSC_ZERO_POINT = 27.0

AION_SPECTRUM_ORIGIN = 3500.0
AION_SPECTRUM_PIXEL = 0.8
AION_SPECTRUM_GRID = AION_SPECTRUM_ORIGIN + AION_SPECTRUM_PIXEL * np.arange(8704)
AION_SPECTRUM_TOKEN_WIDTH = 32 * AION_SPECTRUM_PIXEL
SPECTRUM_SMOOTHING_SIGMA = 2
SPECTRUM_SURVEY = "desi"

AION_EMBEDDING_DIM = 768


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
TABLE_VALUE_COLUMNS = (*LS_TABLE_COLUMNS, *HSC_TABLE_COLUMNS)
N_TABLE_VALUES = len(TABLE_VALUE_COLUMNS) + 1

REDSHIFT = "redshift"
REDSHIFT_TABLE_VALUE = 0

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
STORE_COLUMNS = (*TOKEN_SURVEYS, REDSHIFT)

Catalogue = Literal["ls", "hsc", "desi", "sdss", "gz10", "provabgs"]
CATALOGUES: dict[str, Catalogue] = {
    DATASET_LS_SUFFIX: "ls",
    DATASET_HSC_SUFFIX: "hsc",
    DATASET_DESI_SUFFIX: "desi",
    DATASET_SDSS_SUFFIX: "sdss",
    DATASET_GZ10_SUFFIX: "gz10",
    DATASET_PROVABGS_SUFFIX: "provabgs",
}

WEMM = "tencent/WeMM-Embedding-9B"
WEMM_DIM = 1024
QUERY_PREFIX = "Search galaxy spectra, images and catalogue measurements for: {}"
TextModality = Literal["image", "spectrum"]
PLOT_INCHES = (16.8, 3.84)
TEXT_SURVEYS: dict[TextModality, str] = {
    "image": "ls",
    "spectrum": SPECTRUM_SURVEY,
}


STORES = ("encoded", "tokens")


def store_schema(role: str) -> pa.Schema:
    cell = (
        pa.list_(pa.uint32())
        if role == "tokens"
        else pa.list_(pa.list_(pa.float16(), AION_EMBEDDING_DIM))
    )
    return pa.schema(
        [pa.field("galaxy", pa.int32())]
        + [pa.field(survey, cell) for survey in STORE_COLUMNS]
        + [pa.field(survey, pa.bool_()) for survey in FLAG_SURVEYS]
    )


def store_writer(role: str) -> pa.ipc.RecordBatchFileWriter:
    return pa.ipc.new_file(artifact(role), store_schema(role))


PAIRS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("aion", pa.list_(pa.float32(), AION_EMBEDDING_DIM)),
        pa.field("wemm", pa.list_(pa.float32(), WEMM_DIM)),
    ]
)


TEXT_VIEWS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("view_type", pa.int8()),
        pa.field("round", pa.int16()),
        pa.field("embedding", pa.list_(pa.float16(), WEMM_DIM)),
    ]
)
TEXT_VIEWS_HELD_OUT = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("view_type", pa.int8()),
        pa.field("embedding", pa.list_(pa.float16(), WEMM_DIM)),
    ]
)

TOP_CODES = 64
KEPT = 16
PROBABILITY_FLOOR = float(np.finfo(np.float32).tiny)
SALIENCY_SAMPLE = 512
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
OBSERVATIONS = {
    "ls": "Legacy Survey match",
    "hsc": "HSC match",
    "desi": "DESI spectrum",
    "sdss": "SDSS spectrum",
    REDSHIFT: "usable redshift",
}

MODES = (*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES)
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

ModeName = Literal[*MODES]
TOKEN_GRAMS = {
    mode: (mode, f"{mode}_anywhere") if mode in get_args(TokenMode) else (mode,)
    for mode in MODES
}
GRAM_NAMES = tuple(name for names in TOKEN_GRAMS.values() for name in names)
MAPS: dict[str, frozenset[str]] = {
    "same_place": frozenset(),
    "any_position": frozenset(get_args(TokenMode)),
}
Projection = Literal[*MAPS, "mean", "full"]
TOKEN_EMBEDDING_RANK = 1024
SEARCH_RANK = 256

ARTIFACTS: dict[str, str] = {
    "encoded": "arrow",
    "tokens": "arrow",
    "mean_points": "parquet",
    "full_points": "parquet",
    "parametric_umap": "pt",
    **{f"pairs_{modality}": "parquet" for modality in TEXT_SURVEYS},
    "text_views": "parquet",
    "text_views_held_out": "parquet",
    "text_bases": "npz",
    "text_regression": "pt",
    "text_vectors": "npy",
    "text_query_mean": "npy",
    "probabilities": "arrow",
    "probability_basis": "npz",
    **{f"token_gram_{name}": "npy" for name in GRAM_NAMES},
    **{f"token_embedding_{name}": "npz" for name in GRAM_NAMES},
    **{f"token_search_{name}": "npz" for name in GRAM_NAMES},
    **{f"aion_prediction_{mode}": "npy" for mode in MODES},
    **{f"data_gram_{name}": "npy" for name in GRAM_NAMES},
    **{f"aion_prediction_gram_{name}": "npy" for name in GRAM_NAMES},
    "observed_reference": "npz",
    **{f"aion_prediction_embedding_{name}": "npz" for name in GRAM_NAMES},
    "joint_embedding": "npy",
    "properties": "parquet",
    **{f"{name}_points": "parquet" for name in MAPS},
}

PREDICTIONS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field(
            REDSHIFT, pa.list_(pa.float32(), MODALITY_INFO["tok_z"]["vocab_size"])
        ),
    ]
    + [
        field
        for survey, columns in (("ls", LS_TABLE_COLUMNS), ("hsc", HSC_TABLE_COLUMNS))
        for field in (
            pa.field(
                f"{survey}_codes",
                pa.list_(pa.uint16(), LegacySurveyImage.num_tokens * KEPT),
            ),
            pa.field(
                f"{survey}_probabilities",
                pa.list_(pa.float32(), LegacySurveyImage.num_tokens * KEPT),
            ),
            pa.field(
                f"{survey}_rest", pa.list_(pa.float32(), LegacySurveyImage.num_tokens)
            ),
            pa.field(
                f"{survey}_query_codes",
                pa.list_(
                    pa.uint16(), LegacySurveyImage.num_tokens * (TOP_CODES - KEPT)
                ),
            ),
            pa.field(
                f"{survey}_query_probabilities",
                pa.list_(
                    pa.float16(), LegacySurveyImage.num_tokens * (TOP_CODES - KEPT)
                ),
            ),
            pa.field(
                f"{survey}_tails", pa.list_(pa.float32(), LegacySurveyImage.num_tokens)
            ),
            pa.field(
                f"{survey}_table_values",
                pa.list_(
                    pa.float32(),
                    len(columns) * MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
                ),
            ),
        )
    ]
    + [
        field
        for survey in get_args(SpectrumSurvey)
        for field in (
            pa.field(
                f"{survey}_coefficients",
                pa.list_(
                    pa.uint8(), (DESISpectrum.num_tokens - 1) * SPECTRUM_TOKEN_RANK
                ),
            ),
            pa.field(
                f"{survey}_offsets", pa.list_(pa.float32(), DESISpectrum.num_tokens - 1)
            ),
            pa.field(
                f"{survey}_steps", pa.list_(pa.float32(), DESISpectrum.num_tokens - 1)
            ),
        )
    ]
)


PROVABGS_PROPERTIES = ("mass", "sfr", "ssfr")
LS_PROPERTIES = ("radius", "r_magnitude", "axis_ratio")
SPECTRAL_PROPERTIES = ("halpha_ew", "dn4000")
PROBED = (*PROVABGS_PROPERTIES, *LS_PROPERTIES, *SPECTRAL_PROPERTIES)
PROPERTIES = pa.schema(
    [
        pa.field("galaxy", pa.int32(), nullable=False),
        *(
            pa.field(f"{label}_{part}", pa.float32())
            for label in (REDSHIFT, *PROBED)
            for part in ("measured", "predicted", "width")
        ),
        pa.field("g_r", pa.float32()),
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
    return Path(
        os.environ.get(
            "ALPHAUNIVERSE_CACHE", Path(__file__).resolve().parent.parent / ".cache"
        )
    )


def dataset_dir() -> Path:
    return build_dir() / "dataset"


def artifact(role: str) -> Path:
    return build_dir() / f"{role}.{ARTIFACTS[role]}"


@cache
def galaxy_count() -> int:
    return pq.read_metadata(artifact("mean_points")).num_rows


GalaxyIndex = Annotated[
    int, Field(ge=0), Predicate(lambda galaxy: galaxy < galaxy_count())
]


WANDB_ENTITY = "aistrophysics"
WANDB_PROJECT = "alphaUniverse"
WANDB_MODE = os.environ.get(
    "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled"
)

SEED = 42
