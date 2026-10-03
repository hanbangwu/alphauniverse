from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as F
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    HSCAG,
    HSCAI,
    HSCAR,
    HSCAY,
    HSCAZ,
    DESISpectrum,
    HSCImage,
    HSCMagG,
    HSCMagI,
    HSCMagR,
    HSCMagY,
    HSCMagZ,
    HSCShape11,
    HSCShape12,
    HSCShape22,
    Image,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyFluxI,
    LegacySurveyFluxR,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyFluxZ,
    LegacySurveyImage,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyShapeR,
    Scalar,
    SDSSSpectrum,
    Spectrum,
)
from tqdm import tqdm

from .config import (
    ANCHOR,
    CROP_PIXELS,
    FLAG_SURVEYS,
    HSC,
    LS,
    STORES,
    TOKEN_SURVEYS,
    artifact,
    device,
    store_schema,
)
from .dataset import dataset

LS_SCALARS = (
    (LegacySurveyEBV, f"EBV{LS}"),
    (LegacySurveyFluxG, f"FLUX_G{LS}"),
    (LegacySurveyFluxR, f"FLUX_R{LS}"),
    (LegacySurveyFluxI, f"FLUX_I{LS}"),
    (LegacySurveyFluxZ, f"FLUX_Z{LS}"),
    (LegacySurveyFluxW1, f"FLUX_W1{LS}"),
    (LegacySurveyFluxW2, f"FLUX_W2{LS}"),
    (LegacySurveyFluxW3, f"FLUX_W3{LS}"),
    (LegacySurveyFluxW4, f"FLUX_W4{LS}"),
    (LegacySurveyShapeR, f"SHAPE_R{LS}"),
    (LegacySurveyShapeE1, f"SHAPE_E1{LS}"),
    (LegacySurveyShapeE2, f"SHAPE_E2{LS}"),
)

HSC_SCALARS = (
    (HSCAG, f"a_g{HSC}"),
    (HSCAR, f"a_r{HSC}"),
    (HSCAI, f"a_i{HSC}"),
    (HSCAZ, f"a_z{HSC}"),
    (HSCAY, f"a_y{HSC}"),
    (HSCMagG, f"g_cmodel_mag{HSC}"),
    (HSCMagR, f"r_cmodel_mag{HSC}"),
    (HSCMagI, f"i_cmodel_mag{HSC}"),
    (HSCMagZ, f"z_cmodel_mag{HSC}"),
    (HSCMagY, f"y_cmodel_mag{HSC}"),
    (HSCShape11, f"i_sdssshape_shape11{HSC}"),
    (HSCShape22, f"i_sdssshape_shape22{HSC}"),
    (HSCShape12, f"i_sdssshape_shape12{HSC}"),
)


def codec() -> CodecManager:
    return CodecManager(device=device())


@cache
def model() -> AION:
    return AION.from_pretrained("polymathic-ai/aion-base").to(device()).eval()


def image(
    modality: type[Image], row: dict[str, list], bands: list[str]
) -> torch.Tensor:
    by_band = {
        band.upper(): flux for band, flux in zip(row["band"], row["flux"], strict=True)
    }
    flux = np.asarray([[by_band[band] for band in bands]], dtype=np.float32)
    crop = F.center_crop(torch.from_numpy(flux), output_size=[CROP_PIXELS, CROP_PIXELS])
    return (
        codec()
        .encode(modality(flux=crop.to(device()), bands=bands))[modality.token_key]
        .reshape(1, -1)
    )


def spectrum(modality: type[Spectrum], row: dict[str, list]) -> torch.Tensor:
    fields = {
        "flux": ("flux", torch.float32),
        "ivar": ("ivar", torch.float32),
        "wavelength": ("lambda", torch.float32),
        "mask": ("mask", torch.bool),
    }
    samples = {
        argument: torch.as_tensor(
            np.asarray([row[field]]), dtype=dtype, device=device()
        )
        for argument, (field, dtype) in fields.items()
    }
    return codec().encode(modality(**samples))[modality.token_key].reshape(1, -1)


def scalar(modality: type[Scalar], value: float) -> torch.Tensor:
    return (
        codec()
        .encode(
            modality(
                value=torch.as_tensor([value], dtype=torch.float32, device=device())
            )
        )[modality.token_key]
        .reshape(1, -1)
    )


@torch.inference_mode()
def tokenize(row: dict) -> dict[str, dict[str, torch.Tensor]]:
    groups = {
        ANCHOR: {
            LegacySurveyImage.token_key: image(
                LegacySurveyImage,
                row[TOKEN_SURVEYS[ANCHOR]],
                ["DES-G", "DES-R", "DES-I", "DES-Z"],
            ),
            **{
                modality.token_key: scalar(modality, row[column])
                for modality, column in LS_SCALARS
            },
        }
    }
    if row[TOKEN_SURVEYS["hsc"]] is not None:
        groups["hsc"] = {
            HSCImage.token_key: image(
                HSCImage,
                row[TOKEN_SURVEYS["hsc"]],
                ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            ),
            **{
                modality.token_key: scalar(modality, row[column])
                for modality, column in HSC_SCALARS
            },
        }
    if row[TOKEN_SURVEYS["desi"]] is not None:
        groups["desi"] = {
            DESISpectrum.token_key: spectrum(DESISpectrum, row[TOKEN_SURVEYS["desi"]])
        }
    if row[TOKEN_SURVEYS["sdss"]] is not None:
        groups["sdss"] = {
            SDSSSpectrum.token_key: spectrum(SDSSSpectrum, row[TOKEN_SURVEYS["sdss"]])
        }
    return groups


@torch.inference_mode()
def encode(
    groups: dict[str, dict[str, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = {key: slot for group in groups.values() for key, slot in group.items()}
    encoder_tokens, encoder_embeddings, encoder_mask, modality_mask = (
        model().embed_inputs(
            tokens,
            num_encoder_tokens=sum(slot.shape[1] for slot in tokens.values()),
        )
    )
    with torch.autocast(device_type=device().type, dtype=torch.float16):
        context = model()._encode(encoder_tokens, encoder_embeddings, encoder_mask)

    return (
        context[0].cpu().numpy(),
        encoder_tokens[0].cpu().numpy(),
        modality_mask[0].cpu().numpy(),
    )


def by_survey(
    values: np.ndarray,
    groups: dict[str, dict[str, torch.Tensor]],
    modality_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        survey: values[
            np.flatnonzero(
                np.isin(
                    modality_mask, [model().modality_info[key]["id"] for key in group]
                )
            )
        ]
        for survey, group in groups.items()
    }


def generate_embeddings() -> None:
    data = dataset()
    schemas = {role: store_schema(role) for role in STORES}
    writers = {
        role: pq.ParquetWriter(artifact(role), schemas[role], compression="zstd")
        for role in STORES
    }
    rows: dict[str, list[dict]] = {role: [] for role in STORES}

    for galaxy in tqdm(range(len(data)), desc="encode"):
        row = data[galaxy]
        groups = tokenize(row)
        context, codebook, modality_mask = encode(groups)
        cells = {
            role: {
                survey: list(vectors.astype(np.float16))
                for survey, vectors in by_survey(values, groups, modality_mask).items()
            }
            for role, values in (("encoded", context), ("codebook", codebook))
        } | {
            "tokens": {
                survey: np.concatenate(
                    [
                        slot.reshape(-1).cpu().numpy().astype(dtype=np.uint32)
                        for slot in group.values()
                    ]
                )
                for survey, group in groups.items()
            },
        }
        flags = {
            survey: row[column] is not None for survey, column in FLAG_SURVEYS.items()
        }
        for role in STORES:
            rows[role].append({"galaxy": galaxy, **cells[role], **flags})
            if len(rows[role]) == 1024:
                writers[role].write_table(
                    pa.Table.from_pylist(rows[role], schema=schemas[role])
                )
                rows[role].clear()

    for role in STORES:
        if rows[role]:
            writers[role].write_table(
                pa.Table.from_pylist(rows[role], schema=schemas[role])
            )
        writers[role].close()
