from functools import cache

import numpy as np
import pyarrow as pa
import torch
from aion.codecs.base import Codec
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import (
    DESISpectrum,
    HSCImage,
    LegacySurveyImage,
    SDSSSpectrum,
    Z,
)
from numpy.lib.format import open_memmap
from tqdm import tqdm

from . import encode, pql
from .config import (
    HSC_BANDS,
    HSC_ZERO_POINT,
    IMAGE_MODES,
    LS_BANDS,
    LS_ZERO_POINT,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_MODES,
    TOP_CODES,
    ImageSurvey,
    artifact,
    device,
)

DECODED_BANDS = (*LS_BANDS, *HSC_BANDS)
CENTRAL = (0.16, 0.84)
MODALITIES = {
    "ls_image": LegacySurveyImage,
    "hsc_image": HSCImage,
    "desi_spectrum": DESISpectrum,
    "sdss_spectrum": SDSSSpectrum,
}
TABLE_MODALITIES = {
    REDSHIFT: (Z,),
    "ls_table": tuple(modality for modality, _ in encode.LS_TABLE_VALUES),
    "hsc_table": tuple(modality for modality, _ in encode.HSC_TABLE_VALUES),
}


@cache
def loaded(modality: type) -> Codec:
    return encode.codec()._load_codec(modality)


def decoder(mode: str) -> Codec:
    return loaded(MODALITIES[mode]).to(device())


def codebook(codec: Codec, vocabulary: int) -> torch.Tensor:
    return codec.quantizer.decode(torch.arange(vocabulary, device=device()))


def image_latents(
    rows: pa.RecordBatch, survey: str, vectors: torch.Tensor
) -> torch.Tensor:
    codes, probabilities, spread = pql.query_image_tokens(rows, survey, slice(None))
    codes = torch.from_numpy(codes.astype(np.int64)).to(device())
    probabilities = torch.from_numpy(probabilities).to(device())
    spread = torch.from_numpy(spread).to(device())[..., None]
    latents = torch.einsum("gsc,gscd->gsd", probabilities - spread, vectors[codes])
    return latents + spread * vectors.sum(dim=0)


def spectrum_latents(
    rows: pa.RecordBatch, survey: str, vectors: torch.Tensor
) -> torch.Tensor:
    mean, directions, coefficients = (
        torch.from_numpy(part).to(device())
        for part in (
            *pql.basis()[survey][:2],
            pql.spectrum_token_coefficients(rows, survey, slice(None)),
        )
    )
    return mean @ vectors + coefficients @ (directions @ vectors)


@torch.inference_mode()
def decoded_images(
    codec: Codec, latents: torch.Tensor, survey: ImageSurvey
) -> np.ndarray:
    decoded = codec._decode(latents, bands=DECODED_BANDS).flux
    channels = [
        DECODED_BANDS.index(band)
        for band in (HSC_BANDS if survey == "hsc" else LS_BANDS)
    ]
    zero_point = HSC_ZERO_POINT if survey == "hsc" else LS_ZERO_POINT
    scale = codec.rescaler.convert_zeropoint(zero_point)
    return (decoded[:, channels] * scale).cpu().numpy()


@torch.inference_mode()
def decoded_spectra(codec: Codec, latents: torch.Tensor) -> np.ndarray:
    return (codec._decode(latents, normalization=None).flux + 1).cpu().numpy()


def image_aion_predictions(
    codec: Codec, rows: pa.RecordBatch, survey: ImageSurvey
) -> np.ndarray:
    latents = image_latents(
        rows, survey, codebook(codec, MODALITY_INFO["tok_image"]["vocab_size"])
    )
    return decoded_images(codec, latents.transpose(1, 2), survey)


def spectrum_aion_predictions(
    codec: Codec, rows: pa.RecordBatch, survey: str
) -> np.ndarray:
    latents = spectrum_latents(
        rows, survey, codebook(codec, MODALITY_INFO["tok_spectrum_desi"]["vocab_size"])
    )
    return decoded_spectra(codec, latents.transpose(1, 2))


def bin_values(mode: str) -> np.ndarray:
    _, _, vocabulary = TABLE_MODES[mode]
    values = np.full((len(TABLE_MODALITIES[mode]), vocabulary), np.nan, np.float32)
    for row, modality in zip(values, TABLE_MODALITIES[mode], strict=True):
        quantizer = loaded(modality).quantizer
        decoded = quantizer.decode(torch.arange(quantizer.codebook_size)).numpy()
        top = np.isfinite(decoded[:-1]).sum()
        row[:top] = decoded[:top]
        row[top] = decoded[top - 1]
    return values


def quantiles(
    probabilities: np.ndarray, values: np.ndarray, share: float | np.ndarray
) -> np.ndarray:
    cumulative = np.where(np.isnan(values), 0, probabilities).cumsum(axis=-1)
    index = (cumulative < share * cumulative[..., -1:]).sum(axis=-1)
    return values[np.arange(len(values)), index]


def drawn(weights: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    cumulative = weights.double().cumsum(-1)
    target = torch.from_numpy(rng.random(weights.shape[:-1])).to(cumulative)
    return torch.searchsorted(
        cumulative / cumulative[..., -1:], target[..., None], right=True
    )[..., 0]


def sampled_image_codes(
    rows: pa.RecordBatch, survey: str, rng: np.random.Generator
) -> torch.Tensor:
    codes, probabilities, spread = pql.query_image_tokens(rows, survey, slice(None))
    codes = torch.from_numpy(codes.astype(np.int64)).to(device())
    tail = spread[..., None] * (MODALITY_INFO["tok_image"]["vocab_size"] - TOP_CODES)
    chosen = drawn(
        torch.from_numpy(np.concatenate((probabilities, tail), axis=-1)).to(device()),
        rng,
    )
    rank = torch.from_numpy(
        rng.integers(
            MODALITY_INFO["tok_image"]["vocab_size"] - TOP_CODES,
            size=tuple(chosen.shape),
        )
    ).to(device())
    shifted = codes.sort(dim=-1).values - torch.arange(TOP_CODES, device=device())
    beyond = rank + (shifted <= rank[..., None]).sum(dim=-1)
    return torch.cat((codes, beyond[..., None]), dim=-1).gather(-1, chosen[..., None])[
        ..., 0
    ]


@torch.inference_mode()
def sampled(rows: pa.RecordBatch, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode in TABLE_MODES:
        probabilities = pql.table_value_probabilities(rows, mode)
        share = rng.random((*probabilities.shape[:-1], 1))
        return quantiles(probabilities, bin_values(mode), share)
    codec = decoder(mode)
    if mode in IMAGE_MODES:
        codes = sampled_image_codes(rows, IMAGE_MODES[mode], rng)
        return decoded_images(codec, codec.quantizer.decode(codes), IMAGE_MODES[mode])
    survey = SPECTRUM_MODES[mode]
    coefficients = pql.spectrum_token_coefficients(rows, survey, slice(None))
    weights = torch.from_numpy(pql.distributions(mode, coefficients)).to(device())
    return decoded_spectra(
        codec, codec.quantizer.decode(drawn(weights.clamp(min=0), rng))
    )


def aion_predictions(rows: pa.RecordBatch) -> dict[str, np.ndarray]:
    return {
        **{
            mode: image_aion_predictions(decoder(mode), rows, survey)
            for mode, survey in IMAGE_MODES.items()
        },
        **{
            mode: spectrum_aion_predictions(decoder(mode), rows, survey)
            for mode, survey in SPECTRUM_MODES.items()
        },
        **{
            mode: quantiles(
                pql.table_value_probabilities(rows, mode), bin_values(mode), 0.5
            )
            for mode in TABLE_MODES
        },
    }


def generate_aion_predictions() -> None:
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    count = pql.predictions().num_rows
    stored, start = {}, 0
    for rows in tqdm(pql.predictions().to_batches(), desc="aion_predictions"):
        for mode, found in aion_predictions(rows).items():
            if mode not in stored:
                stored[mode] = open_memmap(
                    artifact(f"aion_prediction_{mode}"),
                    mode="w+",
                    dtype=np.float32,
                    shape=(count, *found.shape[1:]),
                )
            stored[mode][start : start + rows.num_rows] = found
        start += rows.num_rows
    for found in stored.values():
        found.flush()


def deviation(moments: torch.Tensor) -> np.ndarray:
    means, squares = moments.chunk(2, dim=-1)
    return (squares - means**2).sum(dim=-1).mean(dim=-1).sqrt().cpu().numpy()


def widths(
    rows: pa.RecordBatch, depths: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    found = {}
    for mode, survey in IMAGE_MODES.items():
        vectors = codebook(decoder(mode), MODALITY_INFO["tok_image"]["vocab_size"])
        moments = torch.cat((vectors, vectors**2), dim=-1)
        found[mode] = deviation(image_latents(rows, survey, moments))
    for mode, survey in SPECTRUM_MODES.items():
        vectors = codebook(
            decoder(mode), MODALITY_INFO["tok_spectrum_desi"]["vocab_size"]
        )
        moments = torch.cat((vectors, vectors**2), dim=-1)
        found[mode] = deviation(spectrum_latents(rows, survey, moments))
    for mode in TABLE_MODES:
        probabilities, values = (
            pql.table_value_probabilities(rows, mode),
            bin_values(mode),
        )
        lower, upper = (quantiles(probabilities, values, share) for share in CENTRAL)
        found[mode] = np.sqrt((((upper - lower) / 2 * depths[mode]) ** 2).mean(axis=1))
    return found
