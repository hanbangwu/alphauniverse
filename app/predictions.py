import numpy as np
import torch

from . import encode
from .config import ANCHOR, DIM, N_PATCHES, N_SPANS, device

CHUNK = 128
SEED = 0

IMAGES = {ANCHOR: encode.LegacySurveyImage.token_key, "hsc": encode.HSCImage.token_key}
SCALARS = {
    ANCHOR: tuple(modality.token_key for modality, _ in encode.LS_SCALARS),
    "hsc": tuple(modality.token_key for modality, _ in encode.HSC_SCALARS),
}
SPECTRA = {
    "desi": encode.DESISpectrum.token_key,
    "sdss": encode.SDSSSpectrum.token_key,
}

SPAN_TARGETS = {key: np.arange(1, N_SPANS + 1) for key in SPECTRA.values()}
TARGETS = (
    {key: np.arange(N_PATCHES) for key in IMAGES.values()}
    | {key: np.arange(1) for keys in SCALARS.values() for key in keys}
    | SPAN_TARGETS
)


def inputs(row: dict) -> dict[str, torch.Tensor]:
    tokens = {}
    for survey, image_key in IMAGES.items():
        if row[survey] is None:
            continue
        ids = torch.as_tensor(np.asarray(row[survey], dtype=np.int64))[None]
        tokens[image_key] = ids[:, :N_PATCHES]
        for offset, key in enumerate(SCALARS[survey]):
            tokens[key] = ids[:, N_PATCHES + offset : N_PATCHES + offset + 1]
    for survey, key in SPECTRA.items():
        if row[survey] is not None:
            tokens[key] = torch.as_tensor(np.asarray(row[survey], dtype=np.int64))[None]
    return tokens


@torch.inference_mode()
def context(tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_tokens, encoder_embeddings, encoder_mask, _ = encode.model().embed_inputs(
        tokens, num_encoder_tokens=sum(slot.shape[1] for slot in tokens.values())
    )
    with torch.autocast(device_type=device().type, dtype=torch.float16):
        encoded = encode.model()._encode(
            encoder_tokens, encoder_embeddings, encoder_mask
        )
    return encoded, encoder_mask


@torch.inference_mode()
def predict(
    encoded: torch.Tensor, encoder_mask: torch.Tensor, key: str, positions: np.ndarray
) -> np.ndarray:
    aion = encode.model()
    embedding = aion.decoder_embeddings[key]
    order = torch.randperm(
        len(positions), generator=torch.Generator().manual_seed(SEED)
    )
    targets = torch.as_tensor(positions, device=device())
    states = torch.empty(len(positions), DIM, device=device())
    for start in range(0, len(positions), CHUNK):
        chosen = order[start : start + CHUNK].to(device())
        count = len(chosen)
        with torch.autocast(device_type=device().type, dtype=torch.float16):
            decoded = aion._decode(
                encoded,
                encoder_mask,
                aion.mask_token.expand(1, count, -1),
                embedding.pos_emb[:, targets[chosen]] + embedding.mod_emb,
                torch.zeros(1, count, count, dtype=torch.bool, device=device()),
            )
        states[chosen] = decoded[0].float()
    return torch.log_softmax(embedding.forward_logits(states), dim=-1).cpu().numpy()


def predictions(
    tokens: dict[str, torch.Tensor], targets: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    encoded, encoder_mask = context(tokens)
    return {
        key: predict(encoded, encoder_mask, key, positions)
        for key, positions in targets.items()
    }
