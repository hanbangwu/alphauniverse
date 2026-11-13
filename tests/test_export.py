import numpy as np

from app import pql
from app.config import (
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    IMAGE_GRID,
    LS_TABLE_COLUMNS,
)
from app.export import masked, selection


def test_an_unobserved_galaxy_masks_its_whole_map_and_nothing_else() -> None:
    column = masked(np.ones((3, 2, 2)), np.array([True, False, True]))

    assert column.mask.all(axis=(1, 2)).tolist() == [False, True, False]
    assert not column.mask[[0, 2]].any()


def test_the_selection_names_grid_cells_wavelengths_and_catalogue_values() -> None:
    query = pql.Query.model_construct(
        galaxy=0, ls_image=(IMAGE_GRID + 2,), desi_spectrum=(3,), table_values=(0, 1)
    )

    assert selection(query) == {
        "ls_image": [[1, 2]],
        "desi_spectrum": [
            [
                AION_SPECTRUM_ORIGIN + 3 * AION_SPECTRUM_TOKEN_WIDTH,
                AION_SPECTRUM_ORIGIN + 4 * AION_SPECTRUM_TOKEN_WIDTH,
            ]
        ],
        "table_values": ["redshift", LS_TABLE_COLUMNS[0].partition("-")[0]],
    }
