import numpy as np
from astropy.table import MaskedColumn

from alphauniverse.config import (
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    IMAGE_GRID,
    IMAGE_MODES,
    OBSERVATIONS,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_VALUE_COLUMNS,
)
from alphauniverse.encoding import store as encoding
from server.dataset.dataset import catalog
from server.dataset.export import identities, written
from server.search.queries import ModeSearch, Search, Selection

TABLE_VALUES = {REDSHIFT: REDSHIFT} | {
    column.partition("-")[0]: catalog(column) for column in TABLE_VALUE_COLUMNS
}


def masked(values: np.ndarray, observed: np.ndarray) -> MaskedColumn:
    rows = (~observed).reshape((-1,) + (1,) * (values.ndim - 1))
    return MaskedColumn(
        values.astype(np.float32), mask=np.broadcast_to(rows, values.shape)
    )


def selection(query: Selection) -> dict[str, list]:
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
    galaxies: list[str],
    scores: np.ndarray,
    maps: dict[str, np.ndarray],
    query: Search | ModeSearch,
) -> str:
    whole = isinstance(query, ModeSearch)
    found = encoding.rows(galaxies)
    observed = {column: encoding.observed(column)[found] for column in OBSERVATIONS}
    table = identities(found)
    table["score" if whole else "similarity"] = scores.astype(np.float32)
    for mode, survey in (IMAGE_MODES | SPECTRUM_MODES).items():
        values = maps[mode]
        if mode in IMAGE_MODES:
            values = values.reshape(-1, IMAGE_GRID, IMAGE_GRID)
        table[mode] = masked(values, observed[survey])
    for slot, (name, survey) in enumerate(TABLE_VALUES.items()):
        table[name] = masked(maps["table_values"][:, slot], observed[survey])
    table.meta = {"search": "whole modes" if whole else "patches"}
    if not whole:
        table.meta["selection"] = selection(query)
    return written(table)
