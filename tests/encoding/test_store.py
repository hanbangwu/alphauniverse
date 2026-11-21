from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from aion.fourm.modality_info import MODALITY_INFO

from app.encoding.store import image_lattice, spectrum_lattice
from app.predictions.build import aion_predictions


@pytest.mark.parametrize(
    ("lattice", "mode", "modality"),
    [
        (image_lattice, "ls_image", "tok_image"),
        (spectrum_lattice, "desi_spectrum", "tok_spectrum_desi"),
    ],
)
def test_lattice_is_the_codec_s_codebook(
    tree: Path,
    random_weights: None,
    lattice: Callable[[], np.ndarray],
    mode: str,
    modality: str,
) -> None:
    codebook = aion_predictions.codebook(
        aion_predictions.decoder(mode), MODALITY_INFO[modality]["vocab_size"]
    )

    np.testing.assert_array_equal(lattice(), codebook.cpu().numpy())
