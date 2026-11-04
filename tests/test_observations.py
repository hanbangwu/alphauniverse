from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from datasets import Dataset

from app import observations
from app.config import BANDS, CROP_PIXELS, TABLE_VALUE_SURVEYS, TOKEN_SURVEYS

SIZE = CROP_PIXELS + 8


def test_gridded_spectra_keep_each_spectrum_s_inner_product_with_a_template() -> None:
    grid = 3500 + 0.8 * np.arange(200)
    rng = np.random.default_rng(0)
    on_grid = grid[20:120]
    logarithmic = np.concatenate((np.geomspace(3530, 3655, 60), [-1.0, -1.0]))
    cells = pa.array(
        [
            {
                "lambda": wavelength.tolist(),
                "flux": rng.normal(size=len(wavelength)).tolist(),
                "ivar": np.where(
                    np.arange(len(wavelength)) == 3, 0, rng.random(len(wavelength))
                ).tolist(),
                "mask": (np.arange(len(wavelength)) == 5).tolist(),
            }
            for wavelength in (on_grid, logarithmic, grid[-30:])
        ]
    )
    template = rng.random(len(grid))

    weighted, information, squares = observations.gridded(cells, grid)

    for row in range(len(cells)):
        wavelength, flux, ivar = observations.samples(cells, row)
        shape = np.interp(wavelength, grid, template)
        assert len(flux) < len(cells[row]["flux"])
        np.testing.assert_allclose(weighted[row] @ template, ivar * flux @ shape)
        np.testing.assert_allclose(information[row].sum(), ivar.sum())
        np.testing.assert_allclose(squares[row], ivar @ flux**2)


@pytest.fixture
def images(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    rng = np.random.default_rng(1)
    flux = rng.normal(size=(3, len(BANDS["hsc"]), SIZE, SIZE)).astype(np.float32)
    ivar = rng.random(flux.shape).astype(np.float32)
    ivar[:, :, SIZE // 2, SIZE // 2] = 0
    flux[:, :, SIZE // 2, SIZE // 2] = np.nan
    rows = Dataset.from_dict(
        {
            TOKEN_SURVEYS["hsc"]: [
                None
                if galaxy == 1
                else {
                    "band": [band.lower() for band in BANDS["hsc"]],
                    "flux": flux[galaxy],
                    "ivar": ivar[galaxy],
                }
                for galaxy in range(3)
            ]
        }
    )
    monkeypatch.setattr(observations, "dataset", lambda: rows)
    return flux


def test_images_are_the_centre_crop_of_the_galaxies_observed(
    images: np.ndarray,
) -> None:
    start = (SIZE - CROP_PIXELS) // 2
    crop = slice(start, start + CROP_PIXELS)

    flux, _ = observations.images("hsc", np.array([0, 2]))

    np.testing.assert_array_equal(flux, images[[0, 2]][:, :, crop, crop])
    with pytest.raises(ValueError, match="observed"):
        observations.images("hsc", np.array([0, 1, 2]))


def test_measured_images_give_pixels_without_inverse_variance_no_weight(
    images: np.ndarray,
) -> None:
    weighted, information, squares = observations.measured(
        "hsc_image", np.array([0, 2])
    )

    assert np.isfinite(weighted).all() and np.isfinite(squares).all()
    assert (information == 0).sum() == 2 * len(BANDS["hsc"])
    assert (weighted[information == 0] == 0).all()


def test_table_errors_do_not_depend_on_which_galaxies_are_read(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columns = TABLE_VALUE_SURVEYS["ls"]
    rng = np.random.default_rng(2)
    rows = Dataset.from_dict(
        {column: rng.lognormal(size=12).tolist() for column in columns}
    )
    monkeypatch.setattr(observations, "dataset", lambda: rows)
    galaxies = observations.observed("ls_table")

    values, errors = observations.table_values("ls_table", galaxies)
    _, batch_errors = observations.table_values("ls_table", galaxies[2:5])

    np.testing.assert_array_equal(batch_errors, errors[2:5])
    assert values.shape == errors.shape == (len(galaxies), len(columns))


def test_values_are_taken_in_the_order_the_galaxies_are_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = Dataset.from_dict({"value": [0.0, 1.0, 2.0, 3.0, 4.0]})
    monkeypatch.setattr(observations, "dataset", lambda: rows)

    found = observations.column_values("value", np.array([3, 1, 4]))

    np.testing.assert_array_equal(found, [3.0, 1.0, 4.0])
