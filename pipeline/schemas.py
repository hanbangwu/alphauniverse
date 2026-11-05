from __future__ import annotations

from typing import get_args

import numpy as np
import pyarrow as pa
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage

from core.artifacts import artifact
from core.config import (
    FLAG_SURVEYS,
    HSC_TABLE_COLUMNS,
    KEPT,
    LS_TABLE_COLUMNS,
    REDSHIFT,
    SPECTRUM_TOKEN_RANK,
    TOP_CODES,
    WEMM_DIM,
    SpectrumSurvey,
)

from .config import AION_EMBEDDING_DIM, PROBED, STORE_COLUMNS


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
