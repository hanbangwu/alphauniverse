from functools import cache

import numpy as np
import torch
from aion.codecs.quantizers import FiniteScalarQuantizer, LucidrainsLFQ
from aion.fourm.modality_info import MODALITY_INFO

IMAGE_CODE_LEVELS = [7, 5, 5, 5, 5]
SPECTRUM_CODE_DIM = 10


@cache
def image_lattice() -> np.ndarray:
    codes = torch.arange(MODALITY_INFO["tok_image"]["vocab_size"])
    return FiniteScalarQuantizer(levels=IMAGE_CODE_LEVELS).decode(codes).numpy()


@cache
def spectrum_lattice() -> np.ndarray:
    codes = torch.arange(MODALITY_INFO["tok_spectrum_desi"]["vocab_size"])
    quantizer = LucidrainsLFQ(dim=SPECTRUM_CODE_DIM, codebook_size=len(codes))
    return quantizer.decode(codes).numpy()


def false_colors(latents: np.ndarray) -> np.ndarray:
    centered = latents - latents.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    components = centered @ directions[:3].T
    low, high = components.min(axis=0), components.max(axis=0)
    scaled = np.divide(
        components - low,
        high - low,
        out=np.zeros_like(components),
        where=high > low,
    )
    return np.round(scaled * 255).astype(np.uint8)
