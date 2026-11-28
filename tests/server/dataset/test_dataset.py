import numpy as np
import pytest
from datasets import Dataset, Features, Sequence, Value

from alphauniverse.config import (
    AION_SPECTRUM_GRID,
    DATASET_LS_SUFFIX,
    TABLE_VALUE_COLUMNS,
    TOKEN_SURVEYS,
)
from server.dataset import dataset as dataset_module
from server.dataset.dataset import fitted, table, table_columns


def test_fitted_amplitude_weights_unmasked_samples_by_inverse_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = np.linspace(1.0, 2.0, len(AION_SPECTRUM_GRID))
    at = 2000
    cell = {
        "lambda": [0.0, AION_SPECTRUM_GRID[at], AION_SPECTRUM_GRID[at], 6000.0, 7000.0],
        "flux": [1e6, 2 * shape[at], 6 * shape[at], 1e6, 1e6],
        "ivar": [1.0, 3.0, 1.0, 0.0, 1.0],
        "mask": [False, False, False, False, True],
    }
    rows = Dataset.from_dict(
        {TOKEN_SURVEYS["desi"]: [cell]},
        features=Features(
            {
                TOKEN_SURVEYS["desi"]: {
                    name: Sequence(Value("bool" if name == "mask" else "float32"))
                    for name in cell
                }
            }
        ),
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)

    np.testing.assert_allclose(fitted(shape, 0, "desi"), 3 * shape, rtol=1e-6)


def test_table_keeps_numeric_catalog_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = Dataset.from_dict(
        {
            TABLE_VALUE_COLUMNS[0]: [0.5, None],
            f"object_id{DATASET_LS_SUFFIX}": ["a", "b"],
            "_healpix_29": [1, 2],
            TABLE_VALUE_COLUMNS[1]: [None, 3.0],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    table_columns.cache_clear()

    assert table(1) == {TABLE_VALUE_COLUMNS[0]: None, TABLE_VALUE_COLUMNS[1]: 3.0}
    table_columns.cache_clear()
