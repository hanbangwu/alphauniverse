from io import BytesIO

import numpy as np
import pyarrow as pa
import pytest
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

from app import dataset as dataset_module
from app.config import (
    CROP_PIXELS,
    RGB_COLUMN,
    SPECTRUM_SURVEYS,
)
from app.dataset import encode, image, samples, spectrum

SOURCE = Image.fromarray(
    np.random.default_rng(0).integers(
        256, size=(CROP_PIXELS + 8, CROP_PIXELS + 8, 3), dtype=np.uint8
    )
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
            RGB_COLUMN: [SOURCE, SOURCE],
            SPECTRUM_SURVEYS["desi"]: [None, SPECTRUM_CELL],
            SPECTRUM_SURVEYS["sdss"]: [SPECTRUM_CELL, None],
        },
        features=Features(
            {
                RGB_COLUMN: ImageFeature(),
                SPECTRUM_SURVEYS["desi"]: spectrum_feature,
                SPECTRUM_SURVEYS["sdss"]: spectrum_feature,
            }
        ),
    )
    monkeypatch.setattr(dataset_module, "dataset", lambda: rows)


def test_a_rectangular_source_is_cropped_about_its_centre() -> None:
    width, height = CROP_PIXELS + 104, CROP_PIXELS + 8
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - CROP_PIXELS) // 2, (height - CROP_PIXELS) // 2
    pixels[top, left] = (255, 0, 0)
    pixels[top + CROP_PIXELS - 1, left + CROP_PIXELS - 1] = (0, 0, 255)

    with Image.open(BytesIO(encode(Image.fromarray(pixels)))) as png:
        assert png.size == (CROP_PIXELS, CROP_PIXELS)
        assert png.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert png.convert("RGB").getpixel((CROP_PIXELS - 1, CROP_PIXELS - 1)) == (
            0,
            0,
            255,
        )


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


def test_image_decodes_the_stored_rgb_column(stored: None) -> None:
    assert image(1) == encode(SOURCE)


def test_unmatched_survey_has_no_spectrum(stored: None) -> None:
    assert spectrum(0, "desi") is None
    assert spectrum(0, "sdss").column("wavelength").to_pylist() == [4000.0]
