from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from app import pql, predictions, templates
from app.config import (
    ANCHOR,
    BANDS,
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    artifact,
    device,
)

GALAXY = 7


def test_image_latents_average_the_codebook_over_each_cell_s_stored_distribution(
    tree: Path, random_weights: None
) -> None:
    rows = pql.row(GALAXY)
    for mode, survey in IMAGE_MODES.items():
        vectors = templates.codebook(templates.decoder(mode), IMAGE_VOCABULARY)
        dense = torch.from_numpy(pql.dense_image_tokens(rows, survey, slice(None)))

        found = templates.image_latents(rows, survey, vectors)

        np.testing.assert_allclose(
            found.cpu().numpy(),
            (dense.to(vectors) @ vectors).transpose(1, 2).cpu().numpy(),
            atol=1e-5,
            err_msg=mode,
        )


def test_an_image_template_of_certain_codes_is_the_codec_s_decoding_of_them(
    tree: Path, random_weights: None
) -> None:
    codes = np.random.default_rng(0).integers(IMAGE_VOCABULARY, size=N_IMAGE_TOKENS)
    certain = np.full((N_IMAGE_TOKENS, IMAGE_VOCABULARY), -np.inf)
    certain[np.arange(N_IMAGE_TOKENS), codes] = 0
    rows = pql.row(GALAXY)
    for name, values in predictions.image_token_codes(certain).items():
        field = rows.schema.field(f"{ANCHOR}_{name}")
        rows = rows.set_column(
            rows.schema.get_field_index(field.name),
            field,
            pa.FixedSizeListArray.from_arrays(
                pa.array(values.reshape(-1), field.type.value_type),
                field.type.list_size,
            ),
        )
    image = templates.decoder("ls_image")

    found = templates.image_templates(image, rows, ANCHOR)

    with torch.inference_mode():
        expected = image.decode(
            torch.from_numpy(codes)[None].to(device()), bands=BANDS[ANCHOR]
        ).flux
    np.testing.assert_allclose(found, expected.cpu().numpy(), rtol=1e-5, atol=1e-6)


def test_each_stored_template_is_its_galaxy_s_in_every_mode(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")

    templates.generate_templates()

    expected = templates.templates(pql.row(GALAXY))
    for mode, found in expected.items():
        stored = np.load(artifact(f"template_{mode}"))
        assert len(stored) == pql.predictions().num_rows
        assert np.isfinite(stored).all(), mode
        np.testing.assert_allclose(
            stored[GALAXY], found[0], rtol=1e-5, atol=1e-5, err_msg=mode
        )


def test_medians_skip_bins_without_a_value() -> None:
    values = np.array([[1.0, 3.0, np.nan, 5.0]], np.float32)
    probabilities = np.array([[[0.2, 0.2, 0.5, 0.1]]], np.float32)

    found = templates.medians(probabilities, values)

    np.testing.assert_array_equal(found, [[3.0]])
