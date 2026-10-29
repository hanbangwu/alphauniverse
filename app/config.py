from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

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
N_IMAGE_TOKENS = GRID**2

SPECTRUM_ORIGIN = 3500.0
SPECTRUM_PIXEL = 0.8
SPECTRUM_GRID = SPECTRUM_ORIGIN + SPECTRUM_PIXEL * np.arange(8704)
SPECTRUM_SMOOTHING_SIGMA = 2
SPECTRUM_SURVEY: SpectrumSurvey = "desi"
SPECTRUM_TOKEN_WIDTH = 32 * SPECTRUM_PIXEL
N_SPECTRUM_TOKENS = len(SPECTRUM_GRID) // 32

ANCHOR = "ls"
LS = "-mmu_legacysurvey_dr10_south_21"
HSC = "-mmu_hsc_pdr3_dud_22.5"
DESI = "-mmu_desi_edr_sv3"
SDSS = "-mmu_sdss_sdss"
GZ10 = "-mmu_gz10"
PROVABGS = "-mmu_desi_provabgs"


SpectrumSurvey = Literal["desi", "sdss"]
ImageSurvey = Literal["ls", "hsc"]
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
BANDS: dict[str, list[str]] = {
    ANCHOR: ["DES-G", "DES-R", "DES-I", "DES-Z"],
    "hsc": ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
}

TABLE_VALUE_SURVEYS: dict[str, tuple[str, ...]] = {
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
TABLE_VALUE_COLUMNS = tuple(
    column for columns in TABLE_VALUE_SURVEYS.values() for column in columns
)
REDSHIFT = "redshift"
REDSHIFT_TABLE_VALUE = 0
N_TABLE_VALUES = len(TABLE_VALUE_COLUMNS) + 1
REDSHIFT_LIMIT = 6.0
REDSHIFT_COLUMNS: dict[SpectrumSurvey, tuple[str, str, str]] = {
    "desi": (f"Z{DESI}", f"ZERR{DESI}", f"ZWARN{DESI}"),
    "sdss": (f"Z{SDSS}", f"Z_ERR{SDSS}", f"ZWARNING{SDSS}"),
}
STORE_COLUMNS = (*TOKEN_SURVEYS, REDSHIFT)

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
TextModality = Literal["image", "spectrum"]
PLOT_INCHES = (16.8, 3.84)
TEXT_SURVEYS: dict[TextModality, str] = {"image": ANCHOR, "spectrum": SPECTRUM_SURVEY}

Download = Literal["encoded", "codebook", "tokens", "properties"]


STORES = ("encoded", "codebook", "tokens")


def store_schema(role: str) -> pa.Schema:
    cell = (
        pa.list_(pa.uint32())
        if role == "tokens"
        else pa.list_(pa.list_(pa.float16(), DIM))
    )
    return pa.schema(
        [pa.field("galaxy", pa.int32())]
        + [pa.field(survey, cell) for survey in STORE_COLUMNS]
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


TEXT_VIEWS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("view_type", pa.int8()),
        pa.field("round", pa.int16()),
        pa.field("embedding", pa.list_(pa.float16(), GEMMA_DIM)),
    ]
)
TEXT_VIEWS_HELD_OUT = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field("view_type", pa.int8()),
        pa.field("embedding", pa.list_(pa.float16(), GEMMA_DIM)),
    ]
)

TOP_CODES = 64
KEPT = 16
PROBABILITY_FLOOR = float(np.finfo(np.float32).tiny)
SALIENCY_SAMPLE = 512
SPECTRUM_TOKEN_RANK = 256
IMAGE_VOCABULARY = 4375
VOCABULARY = 1024
REDSHIFT_VOCABULARY = 1025
IMAGE_MODES = {f"{survey}_image": survey for survey in get_args(ImageSurvey)}
SPECTRUM_MODES = {f"{survey}_spectrum": survey for survey in SPECTRUM_SURVEYS}
TokenMode = Literal[*IMAGE_MODES, *SPECTRUM_MODES]
TABLE_MODES = {REDSHIFT: (REDSHIFT, 1, REDSHIFT_VOCABULARY)} | {
    f"{survey}_table": (f"{survey}_table_values", len(columns), VOCABULARY)
    for survey, columns in TABLE_VALUE_SURVEYS.items()
}
MODE_SURVEYS = (
    IMAGE_MODES
    | SPECTRUM_MODES
    | {f"{survey}_table": survey for survey in TABLE_VALUE_SURVEYS}
    | {REDSHIFT: REDSHIFT}
)
OBSERVATIONS = {
    "hsc": "HSC match",
    "desi": "DESI spectrum",
    "sdss": "SDSS spectrum",
    REDSHIFT: "usable redshift",
}

MODES = (*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES)
VIEW_KINDS = ("observed", "template", "sample")
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
TEXT_RANK = 32
TEXT_CONDITION = 64
TEXT_POINTS = 16
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
    "codebook": "parquet",
    "tokens": "arrow",
    "mean_points": "parquet",
    "full_points": "parquet",
    "parametric_umap": "pt",
    **{f"pairs_{modality}": "parquet" for modality in TEXT_SURVEYS},
    "alignment": "pt",
    "aion_gemma_space": "npz",
    "text_views": "parquet",
    "text_views_held_out": "parquet",
    "text_bases": "npz",
    "text_flow": "pt",
    "text_clouds": "npy",
    "text_query_mean": "npy",
    "probabilities": "arrow",
    "probability_basis": "npz",
    **{f"token_gram_{name}": "npy" for name in GRAM_NAMES},
    **{f"token_embedding_{name}": "npz" for name in GRAM_NAMES},
    **{f"token_search_{name}": "npz" for name in GRAM_NAMES},
    **{f"template_{mode}": "npy" for mode in MODES},
    **{f"data_gram_{name}": "npy" for name in GRAM_NAMES},
    **{f"template_gram_{name}": "npy" for name in GRAM_NAMES},
    "observed_reference": "npz",
    **{f"template_embedding_{name}": "npz" for name in GRAM_NAMES},
    "joint_embedding": "npy",
    "properties": "parquet",
    **{f"{name}_points": "parquet" for name in MAPS},
}

PREDICTIONS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field(REDSHIFT, pa.list_(pa.float32(), REDSHIFT_VOCABULARY)),
    ]
    + [
        field
        for survey, columns in TABLE_VALUE_SURVEYS.items()
        for field in (
            pa.field(f"{survey}_codes", pa.list_(pa.uint16(), N_IMAGE_TOKENS * KEPT)),
            pa.field(
                f"{survey}_probabilities", pa.list_(pa.float32(), N_IMAGE_TOKENS * KEPT)
            ),
            pa.field(f"{survey}_rest", pa.list_(pa.float32(), N_IMAGE_TOKENS)),
            pa.field(
                f"{survey}_query_codes",
                pa.list_(pa.uint16(), N_IMAGE_TOKENS * (TOP_CODES - KEPT)),
            ),
            pa.field(
                f"{survey}_query_probabilities",
                pa.list_(pa.float16(), N_IMAGE_TOKENS * (TOP_CODES - KEPT)),
            ),
            pa.field(f"{survey}_tails", pa.list_(pa.float32(), N_IMAGE_TOKENS)),
            pa.field(
                f"{survey}_table_values",
                pa.list_(pa.float32(), len(columns) * VOCABULARY),
            ),
        )
    ]
    + [
        field
        for survey in SPECTRUM_SURVEYS
        for field in (
            pa.field(
                f"{survey}_coefficients",
                pa.list_(pa.uint8(), N_SPECTRUM_TOKENS * SPECTRUM_TOKEN_RANK),
            ),
            pa.field(f"{survey}_offsets", pa.list_(pa.float32(), N_SPECTRUM_TOKENS)),
            pa.field(f"{survey}_steps", pa.list_(pa.float32(), N_SPECTRUM_TOKENS)),
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
