import numpy as np

from alphauniverse.config import (
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    IMAGE_GRID,
    LS_TABLE_COLUMNS,
    REDSHIFT,
)
from server.search.export import masked, selection
from server.search.queries import Search


def test_an_unobserved_galaxy_masks_its_whole_map_and_nothing_else() -> None:
    column = masked(np.ones((3, 2, 2)), np.array([True, False, True]))

    assert column.mask.all(axis=(1, 2)).tolist() == [False, True, False]
    assert not column.mask[[0, 2]].any()


def test_the_selection_names_grid_cells_wavelengths_and_catalog_values() -> None:
    query = Search.model_construct(
        galaxy="fixture-0",
        ls_image=(IMAGE_GRID + 2,),
        desi_spectrum=(3,),
        table_values=(0, 1),
    )

    assert selection(query) == {
        "ls_image": [[1, 2]],
        "desi_spectrum": [
            [
                AION_SPECTRUM_ORIGIN + 3 * AION_SPECTRUM_TOKEN_WIDTH,
                AION_SPECTRUM_ORIGIN + 4 * AION_SPECTRUM_TOKEN_WIDTH,
            ]
        ],
        "table_values": [REDSHIFT, LS_TABLE_COLUMNS[0].partition("-")[0]],
    }
