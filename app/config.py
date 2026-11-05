from __future__ import annotations

from typing import Literal

from core.config import (
    DATASET_DESI_SUFFIX,
    DATASET_GZ10_SUFFIX,
    DATASET_HSC_SUFFIX,
    DATASET_LS_SUFFIX,
    DATASET_PROVABGS_SUFFIX,
    DATASET_SDSS_SUFFIX,
    HSC_TABLE_COLUMNS,
    LS_TABLE_COLUMNS,
    MAPS,
    MODES,
    REDSHIFT,
)

TABLE_VALUE_COLUMNS = (*LS_TABLE_COLUMNS, *HSC_TABLE_COLUMNS)

N_TABLE_VALUES = len(TABLE_VALUE_COLUMNS) + 1

REDSHIFT_TABLE_VALUE = 0

Catalogue = Literal["ls", "hsc", "desi", "sdss", "gz10", "provabgs"]

CATALOGUES: dict[str, Catalogue] = {
    DATASET_LS_SUFFIX: "ls",
    DATASET_HSC_SUFFIX: "hsc",
    DATASET_DESI_SUFFIX: "desi",
    DATASET_SDSS_SUFFIX: "sdss",
    DATASET_GZ10_SUFFIX: "gz10",
    DATASET_PROVABGS_SUFFIX: "provabgs",
}

SALIENCY_SAMPLE = 512

OBSERVATIONS = {
    "ls": "Legacy Survey match",
    "hsc": "HSC match",
    "desi": "DESI spectrum",
    "sdss": "SDSS spectrum",
    REDSHIFT: "usable redshift",
}

ModeName = Literal[*MODES]

Projection = Literal[*MAPS, "mean", "full"]
