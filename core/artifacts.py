from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow.fs import LocalFileSystem

from .config import GRAM_NAMES, MAPS, MODES, TEXT_SURVEYS

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
def observed(column: str) -> np.ndarray:
    return pc.is_valid(tokens().column(column)).to_numpy()


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
def aion_predictions(mode: str) -> np.ndarray:
    return np.load(artifact(f"aion_prediction_{mode}"), mmap_mode="r")
