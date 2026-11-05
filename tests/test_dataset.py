from io import BytesIO

import numpy as np
import pyarrow as pa
import pytest
from aion.fourm.modality_info import MODALITY_INFO
from datasets import Dataset, Features, Sequence, Value
from PIL import Image

from app import dataset as dataset_module
from app.config import (
    AION_SPECTRUM_GRID,
    DATASET_LS_SUFFIX,
    FLAG_SURVEYS,
    REDSHIFT_COLUMNS,
    TABLE_VALUE_COLUMNS,
    TOKEN_SURVEYS,
)
from app.dataset import (
    colour,
    encode,
    fitted,
    image,
    predicted_image,
    redshift,
    samples,
    spectrum,
    table,
    table_columns,
)

SPECTRUM_CELL = {"lambda": [-1.0, 4000.0], "flux": [9.0, 1.0], "mask": [True, False]}


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> None:
    spectrum_feature = {
        "lambda": Sequence(Value("float32")),
        "flux": Sequence(Value("float32")),
        "mask": Sequence(Value("bool")),
    }
    rows = Dataset.from_dict(
        {
            TOKEN_SURVEYS["desi"]: [None, SPECTRUM_CELL],
            TOKEN_SURVEYS["sdss"]: [SPECTRUM_CELL, None],
        },
        features=Features(
            {
                TOKEN_SURVEYS["desi"]: spectrum_feature,
                TOKEN_SURVEYS["sdss"]: spectrum_feature,
            }
        ),
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)


def test_a_rectangular_source_is_cropped_about_its_centre() -> None:
    width, height = (
        MODALITY_INFO["tok_image"]["input_size"] + 104,
        MODALITY_INFO["tok_image"]["input_size"] + 8,
    )
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (
        (width - MODALITY_INFO["tok_image"]["input_size"]) // 2,
        (height - MODALITY_INFO["tok_image"]["input_size"]) // 2,
    )
    pixels[top, left] = (255, 0, 0)
    pixels[
        top + MODALITY_INFO["tok_image"]["input_size"] - 1,
        left + MODALITY_INFO["tok_image"]["input_size"] - 1,
    ] = (0, 0, 255)

    with Image.open(BytesIO(encode(Image.fromarray(pixels)))) as png:
        assert png.size == (
            MODALITY_INFO["tok_image"]["input_size"],
            MODALITY_INFO["tok_image"]["input_size"],
        )
        assert png.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert png.convert("RGB").getpixel(
            (
                MODALITY_INFO["tok_image"]["input_size"] - 1,
                MODALITY_INFO["tok_image"]["input_size"] - 1,
            )
        ) == (
            0,
            0,
            255,
        )


def test_colour_maps_bands_by_letter_g_to_blue_and_z_to_red() -> None:
    half = MODALITY_INFO["tok_image"]["input_size"] // 2
    flux = np.zeros(
        (
            5,
            MODALITY_INFO["tok_image"]["input_size"],
            MODALITY_INFO["tok_image"]["input_size"],
        ),
        dtype=np.float32,
    )
    flux[0, :, half:] = 1
    flux[3, :, :half] = 1
    flux[4] = 1

    with Image.open(BytesIO(colour(flux, ["z", "i", "r", "g", "y"]))) as png:
        left, right = (
            png.getpixel((0, 0)),
            png.getpixel((MODALITY_INFO["tok_image"]["input_size"] - 1, 0)),
        )

    assert left[2] > left[0]
    assert right[0] > right[2]


def test_samples_drop_padding_and_blank_masked_flux() -> None:
    cell = pa.scalar(
        {
            "lambda": [-1.0, 4000.0, 4001.0],
            "flux": [9.0, 1.0, 2.0],
            "mask": [True, False, True],
        }
    )

    kept = samples(cell)

    np.testing.assert_array_equal(kept["wavelength"], [4000.0, 4001.0])
    np.testing.assert_array_equal(kept["flux"], [1.0, np.nan])
    assert kept["flux"].dtype == np.float32


def test_ls_images_colour_their_flux_as_predicted_images_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = MODALITY_INFO["tok_image"]["input_size"]
    flux = np.random.default_rng(0).uniform(0, 0.5, (4, size, size))
    flux = flux.astype(np.float32)
    bands = ["des-g", "des-r", "des-i", "des-z"]
    rows = Dataset.from_dict(
        {TOKEN_SURVEYS["ls"]: [{"band": bands, "flux": flux.tolist()}]}
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)

    assert image(0, "ls") == predicted_image(flux, "ls")


@pytest.mark.parametrize(("band", "channel"), [("g", 2), ("r", 1), ("i", 1), ("z", 0)])
def test_hsc_images_colour_each_filter_in_place_in_the_centre_crop(
    monkeypatch: pytest.MonkeyPatch, band: str, channel: int
) -> None:
    bands = ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"]
    flux = np.zeros(
        (
            len(bands),
            MODALITY_INFO["tok_image"]["input_size"] + 8,
            MODALITY_INFO["tok_image"]["input_size"] + 8,
        ),
        dtype=np.float32,
    )
    flux[bands.index(f"hsc-{band}"), 4, 5] = 100.0
    rows = Dataset.from_dict(
        {TOKEN_SURVEYS["hsc"]: [None, {"band": bands, "flux": flux.tolist()}]}
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)

    assert image(0, "hsc") is None
    with Image.open(BytesIO(image(1, "hsc"))) as png:
        pixels = np.asarray(png.convert("RGB"))
    assert pixels.shape == (
        MODALITY_INFO["tok_image"]["input_size"],
        MODALITY_INFO["tok_image"]["input_size"],
        3,
    )
    brightest = np.unravel_index(pixels.sum(axis=2).argmax(), pixels.shape[:2])
    assert brightest == (0, 1)
    assert pixels[0, 1].argmax() == channel


def test_an_hsc_aion_prediction_is_coloured_from_nanomaggies_as_a_legacy_survey_one() -> (
    None
):
    rng = np.random.default_rng(0)
    flux = rng.uniform(
        0,
        0.5,
        (
            4,
            MODALITY_INFO["tok_image"]["input_size"],
            MODALITY_INFO["tok_image"]["input_size"],
        ),
    )
    hsc = np.concatenate(
        [
            flux * 10**1.8,
            rng.uniform(
                0,
                50,
                (
                    1,
                    MODALITY_INFO["tok_image"]["input_size"],
                    MODALITY_INFO["tok_image"]["input_size"],
                ),
            ),
        ]
    )

    assert predicted_image(hsc, "hsc") == predicted_image(flux, "ls")


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


def test_unmatched_survey_has_no_spectrum(stored: None) -> None:
    assert spectrum(0, "desi") is None
    assert spectrum(0, "sdss").column("wavelength").to_pylist() == [4000.0]


def test_table_keeps_numeric_catalogue_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = Dataset.from_dict(
        {
            TABLE_VALUE_COLUMNS[0]: [0.5, None],
            f"object_id{DATASET_LS_SUFFIX}": ["a", "b"],
            "_healpix_29": [1, 2],
            FLAG_SURVEYS["gz10"]: [None, 3],
        }
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)
    table_columns.cache_clear()

    assert table(1) == {TABLE_VALUE_COLUMNS[0]: None, FLAG_SURVEYS["gz10"]: 3}
    table_columns.cache_clear()


@pytest.mark.parametrize(
    ("desi", "sdss", "expected"),
    [
        ((0.5, False), (0.51, False), "desi"),
        ((0.5, True), (0.51, False), "sdss"),
        ((6.5, False), (0.51, False), "sdss"),
        ((float("nan"), False), (0.51, False), "sdss"),
        ((-0.001, False), (None, None), "desi"),
        ((1.5, False), (None, None), "desi"),
        ((None, None), (0.51, True), None),
    ],
)
def test_aion_gets_the_first_usable_redshift_desi_before_sdss(
    desi: tuple[float | None, bool | None],
    sdss: tuple[float | None, bool | None],
    expected: str | None,
) -> None:
    row = {
        column: value
        for (value_column, _, warning_column), (measured, flagged) in zip(
            REDSHIFT_COLUMNS.values(), (desi, sdss), strict=True
        )
        for column, value in ((value_column, measured), (warning_column, flagged))
    }

    assert redshift(row) == expected
