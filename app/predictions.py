from collections.abc import Iterator
from itertools import batched
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from . import encode
from .config import (
    ANCHOR,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    PREDICTIONS,
    REDSHIFT,
    REDSHIFT_VOCABULARY,
    SEED,
    SPECTRUM_TOKEN_RANK,
    STORE_COLUMNS,
    TOP_CODES,
    VOCABULARY,
    artifact,
    device,
)
from .pql import prediction_batch, save_prediction_basis
from .search import source

CHUNK = 128
LEVELS = np.iinfo(np.uint8).max
BATCH = 256

IMAGES = {ANCHOR: encode.LegacySurveyImage.token_key, "hsc": encode.HSCImage.token_key}
TABLE_VALUES = {
    ANCHOR: tuple(modality.token_key for modality, _ in encode.LS_TABLE_VALUES),
    "hsc": tuple(modality.token_key for modality, _ in encode.HSC_TABLE_VALUES),
}
SPECTRA = {
    "desi": encode.DESISpectrum.token_key,
    "sdss": encode.SDSSSpectrum.token_key,
}
REDSHIFT_KEY = encode.Z.token_key


class Mode(NamedTuple):
    name: str
    keys: tuple[str, ...]
    positions: np.ndarray
    vocabulary: int
    survey: str


MODES = (
    *(
        Mode(
            f"{survey}_image",
            (key,),
            np.arange(N_IMAGE_TOKENS),
            IMAGE_VOCABULARY,
            survey,
        )
        for survey, key in IMAGES.items()
    ),
    *(
        Mode(
            f"{survey}_spectrum",
            (key,),
            np.arange(1, N_SPECTRUM_TOKENS + 1),
            VOCABULARY,
            survey,
        )
        for survey, key in SPECTRA.items()
    ),
    *(
        Mode(f"{survey}_table", keys, np.arange(1), VOCABULARY, survey)
        for survey, keys in TABLE_VALUES.items()
    ),
    Mode(REDSHIFT, (REDSHIFT_KEY,), np.arange(1), REDSHIFT_VOCABULARY, REDSHIFT),
)
SPECTRUM_TOKEN_TARGETS = {
    key: np.arange(1, N_SPECTRUM_TOKENS + 1) for key in SPECTRA.values()
}
TARGETS = {key: mode.positions for mode in MODES for key in mode.keys}


def distributions(predicted: dict[str, np.ndarray], mode: Mode) -> np.ndarray:
    return np.concatenate([predicted[key] for key in mode.keys])


def inputs(row: dict) -> dict[str, torch.Tensor]:
    tokens = {}
    if row[REDSHIFT] is not None:
        tokens[REDSHIFT_KEY] = torch.as_tensor(row[REDSHIFT], dtype=torch.int64)[None]
    for survey, image_key in IMAGES.items():
        if row[survey] is None:
            continue
        ids = torch.as_tensor(row[survey], dtype=torch.int64)[None]
        tokens[image_key] = ids[:, :N_IMAGE_TOKENS]
        for offset, key in enumerate(TABLE_VALUES[survey]):
            tokens[key] = ids[:, N_IMAGE_TOKENS + offset : N_IMAGE_TOKENS + offset + 1]
    for survey, key in SPECTRA.items():
        if row[survey] is not None:
            tokens[key] = torch.as_tensor(row[survey], dtype=torch.int64)[None]
    return tokens


@torch.inference_mode()
def decode(
    encoded: torch.Tensor, encoder_mask: torch.Tensor, targets: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    aion = encode.model()
    orders, queries, blocks, first = {}, [], [], 0
    for key, positions in targets.items():
        embedding = aion.decoder_embeddings[key]
        orders[key] = torch.randperm(
            len(positions), generator=torch.Generator().manual_seed(SEED)
        )
        chosen = torch.as_tensor(positions)[orders[key]].to(device())
        queries.append(embedding.pos_emb[:, chosen] + embedding.mod_emb)
        blocks.append(first + torch.arange(len(positions)) // CHUNK)
        first = int(blocks[-1][-1]) + 1
    block = torch.cat(blocks).to(device())
    with torch.autocast(device_type=device().type, dtype=torch.float16):
        decoded = aion._decode(
            encoded,
            encoder_mask,
            aion.mask_token.expand(1, len(block), -1),
            torch.cat(queries, dim=1),
            (block[:, None] != block[None, :])[None],
        )
    states = decoded[0].float().split([len(order) for order in orders.values()])
    predicted = {}
    for (key, order), state in zip(orders.items(), states, strict=True):
        ordered = torch.empty_like(state)
        ordered[order.to(device())] = state
        logits = aion.decoder_embeddings[key].forward_logits(ordered)
        predicted[key] = torch.log_softmax(logits, dim=-1).cpu().numpy()
    return predicted


def predictions(
    tokens: dict[str, torch.Tensor], targets: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    encoded, _, encoder_mask, _ = encode.context(tokens)
    return decode(encoded, encoder_mask, targets)


def image_token_codes(
    log_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.argpartition(-log_probabilities, TOP_CODES - 1, axis=1)[
        :, :TOP_CODES
    ]
    kept = np.take_along_axis(log_probabilities, candidates, axis=1)
    order = np.argsort(-kept, axis=1)
    codes = np.take_along_axis(candidates, order, axis=1)
    kept = np.take_along_axis(kept, order, axis=1).astype(np.float64)
    mass = np.minimum(np.exp(kept).sum(axis=1), np.nextafter(1.0, 0.0))
    return (
        codes.astype(np.uint16),
        kept.astype(np.float16),
        np.log1p(-mass).astype(np.float32),
    )


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = np.zeros(VOCABULARY)
        self.products = np.zeros((VOCABULARY, VOCABULARY))

    def add(self, probabilities: np.ndarray) -> None:
        rows = probabilities.astype(np.float64)
        self.count += len(rows)
        self.total += rows.sum(axis=0)
        self.products += rows.T @ rows

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total / self.count
        _, vectors = np.linalg.eigh(self.products / self.count - np.outer(mean, mean))
        return mean, vectors[:, ::-1][:, :SPECTRUM_TOKEN_RANK].T


def coefficients(
    probabilities: np.ndarray, mean: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = (probabilities - mean) @ directions.T
    offsets = projected.min(axis=1)
    steps = (
        np.maximum(projected.max(axis=1) - offsets, np.finfo(np.float32).tiny) / LEVELS
    )
    return (
        np.rint((projected - offsets[:, None]) / steps[:, None]).astype(np.uint8),
        offsets.astype(np.float32),
        steps.astype(np.float32),
    )


def rows(description: str) -> Iterator[dict]:
    dataset = source("tokens")
    with tqdm(total=dataset.count_rows(), desc=description) as progress:
        for batch in dataset.to_batches(columns=["galaxy", *STORE_COLUMNS]):
            yield from batch.to_pylist()
            progress.update(batch.num_rows)


def bases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    moments = {survey: Moments() for survey in SPECTRA}
    for row in rows("spectrum token moments"):
        predicted = predictions(inputs(row), SPECTRUM_TOKEN_TARGETS)
        for survey, key in SPECTRA.items():
            moments[survey].add(np.exp(predicted[key]))
    return {survey: moment.basis() for survey, moment in moments.items()}


def record(
    galaxy: int,
    predicted: dict[str, np.ndarray],
    fitted: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "galaxy": np.asarray([galaxy], dtype=np.int32),
        REDSHIFT: predicted[REDSHIFT_KEY][0].astype(np.float16),
    }
    for survey, image_key in IMAGES.items():
        codes, kept, tails = image_token_codes(predicted[image_key])
        values[f"{survey}_codes"] = codes.reshape(-1)
        values[f"{survey}_log_probabilities"] = kept.reshape(-1)
        values[f"{survey}_tails"] = tails
        values[f"{survey}_table_values"] = np.concatenate(
            [predicted[key][0] for key in TABLE_VALUES[survey]]
        ).astype(np.float16)
    for survey, key in SPECTRA.items():
        quantised, offsets, steps = coefficients(
            np.exp(predicted[key]), *fitted[survey]
        )
        values[f"{survey}_coefficients"] = quantised.reshape(-1)
        values[f"{survey}_offsets"] = offsets
        values[f"{survey}_steps"] = steps
    return values


def generate_predictions() -> None:
    fitted = bases()
    save_prediction_basis(fitted)
    with pa.ipc.new_file(artifact("predictions"), PREDICTIONS) as writer:
        for chunk in batched(rows("predict"), BATCH):
            writer.write_batch(
                prediction_batch(
                    [
                        record(row["galaxy"], predictions(inputs(row), TARGETS), fitted)
                        for row in chunk
                    ]
                )
            )
