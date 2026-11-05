from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from aion.codecs.preprocessing.image import RescaleToLegacySurvey
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import LegacySurveyImage

from app import pql
from core import artifacts, stored_predictions
from core.artifacts import artifact
from core.config import (
    HSC_BANDS,
    IMAGE_MODES,
    LS_BANDS,
    MODES,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_MODES,
)
from core.device import device
from pipeline import aion_predictions, predictions

GALAXY = 7


def test_image_latents_average_the_codebook_over_each_cell_s_stored_distribution(
    tree: Path, random_weights: None
) -> None:
    rows = pql.row(GALAXY)
    for mode, survey in IMAGE_MODES.items():
        vectors = aion_predictions.codebook(
            aion_predictions.decoder(mode), MODALITY_INFO["tok_image"]["vocab_size"]
        )
        dense = torch.from_numpy(pql.dense_image_tokens(rows, survey, slice(None)))

        found = aion_predictions.image_latents(rows, survey, vectors)

        np.testing.assert_allclose(
            found.cpu().numpy(),
            (dense.to(vectors.device).double() @ vectors.double()).cpu().numpy(),
            atol=1e-5,
            err_msg=mode,
        )


def certain(codes: np.ndarray) -> pa.RecordBatch:
    logits = np.full(
        (LegacySurveyImage.num_tokens, MODALITY_INFO["tok_image"]["vocab_size"]),
        -np.inf,
    )
    logits[np.arange(LegacySurveyImage.num_tokens), codes] = 0
    rows = pql.row(GALAXY)
    for survey in IMAGE_MODES.values():
        for name, values in predictions.image_token_codes(logits).items():
            field = rows.schema.field(f"{survey}_{name}")
            rows = rows.set_column(
                rows.schema.get_field_index(field.name),
                field,
                pa.FixedSizeListArray.from_arrays(
                    pa.array(values.reshape(-1), field.type.value_type),
                    field.type.list_size,
                ),
            )
    return rows


def test_an_image_aion_prediction_of_certain_codes_is_the_codec_s_decoding_of_them(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RescaleToLegacySurvey,
        "_reverse_zeropoint",
        RescaleToLegacySurvey.convert_zeropoint,
        raising=False,
    )
    codes = np.random.default_rng(0).integers(
        MODALITY_INFO["tok_image"]["vocab_size"], size=LegacySurveyImage.num_tokens
    )
    rows = certain(codes)
    for mode, survey in IMAGE_MODES.items():
        image = aion_predictions.decoder(mode)

        found = aion_predictions.image_aion_predictions(image, rows, survey)

        with torch.inference_mode():
            expected = image.decode(
                torch.from_numpy(codes)[None].to(device()),
                bands=HSC_BANDS if survey == "hsc" else LS_BANDS,
            ).flux
        np.testing.assert_allclose(
            found, expected.cpu().numpy(), rtol=1e-5, atol=1e-6, err_msg=mode
        )


def test_a_spectrum_aion_prediction_decodes_the_codebook_averaged_over_each_window_s_distribution(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    for survey, (_, directions, _, _) in list(stored_predictions.basis().items()):
        mean = rng.dirichlet(
            np.ones(MODALITY_INFO["tok_spectrum_desi"]["vocab_size"])
        ).astype(np.float32)
        monkeypatch.setitem(
            stored_predictions.basis(),
            survey,
            (mean, directions, mean @ mean, directions @ mean),
        )
    rows = pql.row(GALAXY)
    for mode, survey in SPECTRUM_MODES.items():
        spectrum = aion_predictions.decoder(mode)
        vectors = aion_predictions.codebook(
            spectrum, MODALITY_INFO["tok_spectrum_desi"]["vocab_size"]
        )
        coefficients = stored_predictions.spectrum_token_coefficients(
            rows, survey, slice(None)
        )
        dense = torch.from_numpy(
            stored_predictions.distributions(mode, coefficients)
        ).to(vectors)

        found = aion_predictions.spectrum_aion_predictions(spectrum, rows, survey)

        expected = aion_predictions.decoded_spectra(
            spectrum, (dense @ vectors).transpose(1, 2)
        )
        np.testing.assert_allclose(found, expected, rtol=1e-5, atol=1e-5, err_msg=mode)


def test_each_stored_aion_prediction_is_its_galaxy_s_in_every_mode(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")

    aion_predictions.generate_aion_predictions()

    expected = aion_predictions.aion_predictions(pql.row(GALAXY))
    for mode, found in expected.items():
        stored = np.load(artifact(f"aion_prediction_{mode}"))
        assert len(stored) == stored_predictions.predictions().num_rows
        assert np.isfinite(stored).all(), mode
        np.testing.assert_allclose(
            stored[GALAXY], found[0], rtol=1e-5, atol=1e-5, err_msg=mode
        )


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        ([1.0, 2.0, 3.0, 1e9], [1.0, 2.0, 3.0, 3.0]),
        ([1.0, 2.0, np.nan, np.nan], [1.0, 2.0, 2.0, np.nan]),
    ],
)
def test_a_scalar_codec_s_open_top_bin_takes_its_lower_edge(
    decoded: list[float], expected: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    quantizer = SimpleNamespace(
        codebook_size=len(decoded), decode=torch.tensor(decoded).__getitem__
    )
    monkeypatch.setattr(
        aion_predictions, "loaded", lambda _: SimpleNamespace(quantizer=quantizer)
    )

    found = aion_predictions.bin_values(REDSHIFT)

    np.testing.assert_array_equal(found[0, : len(decoded)], expected)
    assert np.isnan(found[0, len(decoded) :]).all()


def test_quantiles_skip_bins_without_a_value() -> None:
    values = np.array([[1.0, 3.0, np.nan, 5.0]], np.float32)
    probabilities = np.array([[[0.2, 0.2, 0.5, 0.1]]], np.float32)

    found = [
        aion_predictions.quantiles(probabilities, values, share)
        for share in (0.16, 0.5, 0.84)
    ]

    np.testing.assert_array_equal(found, [[[1.0]], [[3.0]], [[5.0]]])


def test_widths_are_the_predicted_spread_in_every_mode(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = certain(
        np.random.default_rng(1).integers(
            MODALITY_INFO["tok_image"]["vocab_size"], size=LegacySurveyImage.num_tokens
        )
    )
    rng = np.random.default_rng(2)
    values = {
        mode: np.sort(rng.normal(size=(count, vocabulary)), axis=1)
        for mode, (_, count, vocabulary) in TABLE_MODES.items()
    }
    depths = {mode: rng.random(len(found)) for mode, found in values.items()}
    monkeypatch.setattr(aion_predictions, "bin_values", values.get)

    found = aion_predictions.widths(rows, depths)

    for mode in IMAGE_MODES:
        np.testing.assert_allclose(found[mode], 0, atol=1e-6, err_msg=mode)
    for mode, survey in SPECTRUM_MODES.items():
        vectors = aion_predictions.codebook(
            aion_predictions.decoder(mode),
            MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
        ).double()
        coefficients = stored_predictions.spectrum_token_coefficients(
            rows, survey, slice(None)
        )
        dense = torch.from_numpy(
            stored_predictions.distributions(mode, coefficients)
        ).to(vectors)
        variance = dense @ vectors**2 - (dense @ vectors) ** 2
        np.testing.assert_allclose(
            found[mode],
            variance.sum(-1).mean(-1).sqrt().cpu().numpy(),
            rtol=1e-4,
            err_msg=mode,
        )
    for mode, bins in values.items():
        cumulative = stored_predictions.table_value_probabilities(rows, mode)[0].cumsum(
            axis=-1
        )
        lower, upper = (
            bins[
                np.arange(len(bins)), (cumulative < share * cumulative[:, -1:]).sum(-1)
            ]
            for share in aion_predictions.CENTRAL
        )
        np.testing.assert_allclose(
            found[mode],
            [np.sqrt((((upper - lower) / 2 * depths[mode]) ** 2).mean())],
            rtol=1e-5,
            err_msg=mode,
        )


class Fixed:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.value)


def test_draws_at_either_end_of_the_unit_interval_skip_entries_of_zero_probability() -> (
    None
):
    probabilities = np.array([[0.0, 0.5, 0.0, 0.5], [0.3, 0.7, 0.0, 0.0]], np.float32)

    for value in (0.0, np.nextafter(1.0, 0.0)):
        found = aion_predictions.drawn(torch.from_numpy(probabilities), Fixed(value))

        assert probabilities[np.arange(len(probabilities)), found.numpy()].all()


@pytest.mark.parametrize("survey", IMAGE_MODES.values())
def test_image_codes_drawn_from_the_tail_are_never_a_cell_s_stored_codes(
    tree: Path, monkeypatch: pytest.MonkeyPatch, survey: str
) -> None:
    rows = pql.row(GALAXY)
    codes, probabilities, spread = stored_predictions.query_image_tokens(
        rows, survey, slice(None)
    )
    tail_only = (
        codes,
        0 * probabilities,
        np.full_like(spread, 1 / MODALITY_INFO["tok_image"]["vocab_size"]),
    )
    monkeypatch.setattr(stored_predictions, "query_image_tokens", lambda *_: tail_only)

    found = aion_predictions.sampled_image_codes(rows, survey, np.random.default_rng(0))

    found = found.cpu().numpy()
    assert not (found[..., None] == codes).any()
    assert found.min() >= 0 and found.max() < MODALITY_INFO["tok_image"]["vocab_size"]
    assert len(np.unique(found)) > codes.shape[-1]


def test_a_sample_has_the_shape_of_the_mode_s_aion_prediction(
    tree: Path, random_weights: None
) -> None:
    rows = stored_predictions.predictions().slice(GALAXY, 2).to_batches()[0]
    for mode in MODES:
        found = aion_predictions.sampled(rows, mode, np.random.default_rng(0))

        assert (
            found.shape == artifacts.aion_predictions(mode)[GALAXY : GALAXY + 2].shape
        ), mode
