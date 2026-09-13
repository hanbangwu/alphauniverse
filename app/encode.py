from functools import cache
from pathlib import Path

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
    BUILD_DIR,
    CROP_PX,
    DATASET_ID,
    DATASET_REVISION,
    DIM,
    FLAG_SURVEYS,
    HSC,
    LS,
    TOKEN_SURVEYS,
    artifact,
    device,
)
from .dataset import dataset

STORES = ("encoded", "codebook", "tokens")

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


@cache
def codec() -> CodecManager:
    return CodecManager(device=device())


@cache
def model() -> AION:
    net = AION.from_pretrained("polymathic-ai/aion-base").to(device()).eval()
    net.requires_grad_(False)
    return net


def image(
    modality: type[Image], row: dict[str, list], bands: list[str]
) -> torch.Tensor:
    by_band = {
        band.upper(): flux for band, flux in zip(row["band"], row["flux"], strict=True)
    }
    flux = np.asarray([[by_band[band] for band in bands]], dtype=np.float32)
    crop = F.center_crop(torch.from_numpy(flux), output_size=[CROP_PX, CROP_PX])
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


def tokenize(row: dict) -> dict[str, dict[str, torch.Tensor]]:
    groups = {
        ANCHOR: {
            LegacySurveyImage.token_key: image(
                LegacySurveyImage,
                row[TOKEN_SURVEYS[ANCHOR]],
                ["DES-G", "DES-R", "DES-I", "DES-Z"],
            ),
            **{m.token_key: scalar(m, row[column]) for m, column in LS_SCALARS},
        }
    }
    if row[TOKEN_SURVEYS["hsc"]] is not None:
        groups["hsc"] = {
            HSCImage.token_key: image(
                HSCImage,
                row[TOKEN_SURVEYS["hsc"]],
                ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            ),
            **{m.token_key: scalar(m, row[column]) for m, column in HSC_SCALARS},
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


def encode(
    groups: dict[str, dict[str, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = {key: slot for group in groups.values() for key, slot in group.items()}
    with torch.no_grad():
        enc_tokens, enc_emb, enc_mask, mod_mask = model().embed_inputs(
            tokens, num_encoder_tokens=sum(slot.shape[1] for slot in tokens.values())
        )
        with torch.autocast(device_type=device().type, dtype=torch.float16):
            context = model()._encode(enc_tokens, enc_emb, enc_mask)

    return (
        context[0].cpu().numpy(),
        enc_tokens[0].cpu().numpy(),
        mod_mask[0].cpu().numpy(),
    )


def by_survey(
    values: np.ndarray,
    groups: dict[str, dict[str, torch.Tensor]],
    mod_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        survey: values[
            np.flatnonzero(
                np.isin(mod_mask, [model().modality_info[key]["id"] for key in group])
            )
        ]
        for survey, group in groups.items()
    }


def generate_embeddings() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    data = dataset(DATASET_ID, DATASET_REVISION)
    count = len(data)

    batches: dict[str, dict[str, list]] = {
        role: {key: [] for key in ("galaxy", *TOKEN_SURVEYS, *FLAG_SURVEYS)}
        for role in STORES
    }

    schemas: dict[str, pa.Schema] = {}
    staging: dict[str, Path] = {}
    writers: dict[str, pq.ParquetWriter] = {}
    for role in STORES:
        path = artifact(role)
        path.unlink(missing_ok=True)
        staging[role] = path.with_name(f"{path.name}.partial")
        staging[role].unlink(missing_ok=True)

        cell = (
            pa.list_(pa.uint32())
            if role == "tokens"
            else pa.list_(pa.list_(pa.float16(), DIM))
        )
        schemas[role] = pa.schema(
            [pa.field("galaxy", pa.int32())]
            + [pa.field(survey, cell) for survey in TOKEN_SURVEYS]
            + [pa.field(survey, pa.bool_()) for survey in FLAG_SURVEYS]
        )
        writers[role] = pq.ParquetWriter(
            staging[role], schemas[role], compression="zstd"
        )

    for i in tqdm(range(count), desc="encode"):
        row = data[i]
        groups = tokenize(row)
        context, codebook, mod_mask = encode(groups)
        cells = {
            "encoded": by_survey(context, groups, mod_mask),
            "codebook": by_survey(codebook, groups, mod_mask),
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
            batch = batches[role]
            batch["galaxy"].append(i)
            for survey in TOKEN_SURVEYS:
                batch[survey].append(cells[role].get(survey))
            for survey in FLAG_SURVEYS:
                batch[survey].append(flags[survey])
            if len(batch["galaxy"]) < 1024 and i < count - 1:
                continue

            dtype, dim = (np.uint32, None) if role == "tokens" else (np.float16, DIM)
            columns: dict[str, pa.Array] = {}
            for survey in TOKEN_SURVEYS:
                entries = batch[survey]
                valid = [entry for entry in entries if entry is not None]
                values = pa.array(
                    np.concatenate(valid).reshape(-1) if valid else np.empty(0, dtype)
                )
                offsets = pa.array(
                    np.cumsum(
                        [0, *(0 if entry is None else len(entry) for entry in entries)]
                    ),
                    type=pa.int32(),
                    mask=np.array([*(entry is None for entry in entries), False]),
                )
                columns[survey] = pa.ListArray.from_arrays(
                    offsets,
                    values
                    if dim is None
                    else pa.FixedSizeListArray.from_arrays(values, dim),
                )

            writers[role].write_table(
                pa.table(
                    {
                        "galaxy": pa.array(batch["galaxy"], type=pa.int32()),
                        **columns,
                        **{
                            survey: pa.array(batch[survey], type=pa.bool_())
                            for survey in FLAG_SURVEYS
                        },
                    },
                    schema=schemas[role],
                )
            )
            for values in batch.values():
                values.clear()

    for role in STORES:
        writers[role].close()
        staging[role].replace(artifact(role))
