import numpy as np
import pyarrow as pa
import torch
from aion.codecs.base import Codec
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
    ANCHOR,
    BANDS,
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_MODES,
    VOCABULARY,
    artifact,
    device,
)

DECODED_BANDS = [*BANDS[ANCHOR], *BANDS["hsc"]]
ZERO_POINTS = {ANCHOR: 22.5, "hsc": 27.0}
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


def decoder(mode: str) -> Codec:
    return encode.codec()._load_codec(MODALITIES[mode]).to(device())


def codebook(codec: Codec, vocabulary: int) -> torch.Tensor:
    return codec.quantizer.decode(torch.arange(vocabulary, device=device())).float()


def image_latents(
    rows: pa.RecordBatch, survey: str, vectors: torch.Tensor
) -> torch.Tensor:
    codes, probabilities, spread = pql.query_image_tokens(rows, survey, slice(None))
    codes = torch.from_numpy(codes.astype(np.int64)).to(device())
    probabilities = torch.from_numpy(probabilities).to(device())
    spread = torch.from_numpy(spread).to(device())[..., None]
    latents = torch.einsum("gsc,gscd->gsd", probabilities - spread, vectors[codes])
    return (latents + spread * vectors.sum(dim=0)).transpose(1, 2)


@torch.inference_mode()
def decoded_images(codec: Codec, latents: torch.Tensor, survey: str) -> np.ndarray:
    decoded = codec._decode(latents, bands=DECODED_BANDS).flux
    channels = [DECODED_BANDS.index(band) for band in BANDS[survey]]
    scale = codec.rescaler.convert_zeropoint(ZERO_POINTS[survey])
    return (decoded[:, channels] * scale).cpu().numpy()


@torch.inference_mode()
def decoded_spectra(codec: Codec, latents: torch.Tensor) -> np.ndarray:
    return (codec._decode(latents, normalization=None).flux + 1).cpu().numpy()


def image_templates(codec: Codec, rows: pa.RecordBatch, survey: str) -> np.ndarray:
    vectors = codebook(codec, IMAGE_VOCABULARY)
    return decoded_images(codec, image_latents(rows, survey, vectors), survey)


def spectrum_templates(codec: Codec, rows: pa.RecordBatch, survey: str) -> np.ndarray:
    vectors = codebook(codec, VOCABULARY)
    mean, directions, coefficients = (
        torch.from_numpy(part).to(device())
        for part in (
            *pql.basis()[survey][:2],
            pql.spectrum_token_coefficients(rows, survey, slice(None)),
        )
    )
    latents = mean @ vectors + coefficients @ (directions @ vectors)
    return decoded_spectra(codec, latents.transpose(1, 2))


def bin_values(mode: str) -> np.ndarray:
    _, _, vocabulary = TABLE_MODES[mode]
    values = np.full((len(TABLE_MODALITIES[mode]), vocabulary), np.nan, np.float32)
    for row, modality in zip(values, TABLE_MODALITIES[mode], strict=True):
        quantizer = encode.codec()._load_codec(modality).quantizer
        row[: quantizer.codebook_size] = quantizer.decode(
            torch.arange(quantizer.codebook_size)
        )
    return values


def medians(probabilities: np.ndarray, values: np.ndarray) -> np.ndarray:
    cumulative = np.where(np.isnan(values), 0, probabilities).cumsum(axis=-1)
    index = (cumulative < cumulative[..., -1:] / 2).sum(axis=-1, keepdims=True)
    return np.take_along_axis(
        np.broadcast_to(values, probabilities.shape), index, axis=-1
    )[..., 0]


def templates(
    rows: pa.RecordBatch, decoders: dict[str, Codec], values: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        **{
            mode: image_templates(decoders[mode], rows, survey)
            for mode, survey in IMAGE_MODES.items()
        },
        **{
            mode: spectrum_templates(decoders[mode], rows, survey)
            for mode, survey in SPECTRUM_MODES.items()
        },
        **{
            mode: medians(pql.table_value_probabilities(rows, mode), values[mode])
            for mode in TABLE_MODES
        },
    }


def generate_templates() -> None:
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    decoders = {mode: decoder(mode) for mode in MODALITIES}
    values = {mode: bin_values(mode) for mode in TABLE_MODES}
    count = pql.predictions().num_rows
    stored, start = {}, 0
    for rows in tqdm(pql.predictions().to_batches(), desc="templates"):
        for mode, found in templates(rows, decoders, values).items():
            if mode not in stored:
                stored[mode] = open_memmap(
                    artifact(f"template_{mode}"),
                    mode="w+",
                    dtype=np.float32,
                    shape=(count, *found.shape[1:]),
                )
            stored[mode][start : start + rows.num_rows] = found
        start += rows.num_rows
    for found in stored.values():
        found.flush()
