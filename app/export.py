from io import StringIO

import astropy.units as u
import numpy as np
from astropy.table import MaskedColumn, Table

from . import pql
from .config import (
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    IMAGE_GRID,
    IMAGE_MODES,
    OBSERVATIONS,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_VALUE_COLUMNS,
)
from .dataset import catalogue, identity

TABLE_VALUES = {REDSHIFT: REDSHIFT} | {
    column.partition("-")[0]: catalogue(column) for column in TABLE_VALUE_COLUMNS
}


def identities(galaxies: np.ndarray) -> Table:
    table = Table(rows=[identity(galaxy) for galaxy in galaxies.tolist()])
    table["ra"].unit = table["dec"].unit = u.deg
    return table


def written(table: Table, format: str) -> str:
    text = StringIO()
    table.write(text, format=format)
    return text.getvalue()


def masked(values: np.ndarray, observed: np.ndarray) -> MaskedColumn:
    rows = (~observed).reshape((-1,) + (1,) * (values.ndim - 1))
    return MaskedColumn(
        values.astype(np.float32), mask=np.broadcast_to(rows, values.shape)
    )


def selection(query: pql.Query) -> dict[str, list]:
    images = {
        mode: [list(divmod(slot, IMAGE_GRID)) for slot in slots]
        for mode in IMAGE_MODES
        if (slots := getattr(query, mode))
    }
    spectra = {
        mode: [
            [
                AION_SPECTRUM_ORIGIN + slot * AION_SPECTRUM_TOKEN_WIDTH,
                AION_SPECTRUM_ORIGIN + (slot + 1) * AION_SPECTRUM_TOKEN_WIDTH,
            ]
            for slot in slots
        ]
        for mode in SPECTRUM_MODES
        if (slots := getattr(query, mode))
    }
    names = list(TABLE_VALUES)
    values = (
        {"table_values": [names[slot] for slot in query.table_values]}
        if query.table_values
        else {}
    )
    return images | spectra | values


def search_ecsv(
    galaxies: np.ndarray,
    scores: np.ndarray,
    maps: dict[str, np.ndarray],
    query: pql.Query | pql.ModeQuery,
) -> str:
    whole = isinstance(query, pql.ModeQuery)
    observed = {column: pql.observed(column)[galaxies] for column in OBSERVATIONS}
    table = identities(galaxies)
    table["score" if whole else "similarity"] = scores.astype(np.float32)
    for mode, survey in (IMAGE_MODES | SPECTRUM_MODES).items():
        values = maps[mode]
        if mode in IMAGE_MODES:
            values = values.reshape(-1, IMAGE_GRID, IMAGE_GRID)
        table[mode] = masked(values, observed[survey])
    for slot, (name, survey) in enumerate(TABLE_VALUES.items()):
        table[name] = masked(maps["table_values"][:, slot], observed[survey])
    table.meta = {
        "rows": "The query galaxy, then its matches in rank order",
        "search": "whole modes" if whole else "tokens",
        "request": query.model_dump(mode="json"),
    }
    if not whole:
        table.meta["selection"] = selection(query)
    return written(table, "ascii.ecsv")


def text_search_csv(galaxies: np.ndarray, scores: np.ndarray) -> str:
    table = identities(galaxies)
    table["cosine_similarity"] = scores
    return written(table, "ascii.csv")
