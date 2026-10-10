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
    DIM,
    IMAGE_VOCABULARY,
    N_PATCHES,
    N_SPANS,
    PREDICTIONS,
    REDSHIFT,
    REDSHIFT_VOCABULARY,
    SEED,
    SPAN_RANK,
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
SCALARS = {
    ANCHOR: tuple(modality.token_key for modality, _ in encode.LS_SCALARS),
    "hsc": tuple(modality.token_key for modality, _ in encode.HSC_SCALARS),
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
        Mode(f"{survey}_image", (key,), np.arange(N_PATCHES), IMAGE_VOCABULARY, survey)
        for survey, key in IMAGES.items()
    ),
    *(
        Mode(
            f"{survey}_spectrum", (key,), np.arange(1, N_SPANS + 1), VOCABULARY, survey
        )
        for survey, key in SPECTRA.items()
    ),
    *(
        Mode(f"{survey}_table", keys, np.arange(1), VOCABULARY, survey)
        for survey, keys in SCALARS.items()
    ),
    Mode(REDSHIFT, (REDSHIFT_KEY,), np.arange(1), REDSHIFT_VOCABULARY, REDSHIFT),
)
SPAN_TARGETS = {key: np.arange(1, N_SPANS + 1) for key in SPECTRA.values()}
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
        tokens[image_key] = ids[:, :N_PATCHES]
        for offset, key in enumerate(SCALARS[survey]):
            tokens[key] = ids[:, N_PATCHES + offset : N_PATCHES + offset + 1]
    for survey, key in SPECTRA.items():
        if row[survey] is not None:
            tokens[key] = torch.as_tensor(row[survey], dtype=torch.int64)[None]
    return tokens


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
    encoded, _, encoder_mask, _ = encode.context(tokens)
    return {
        key: predict(encoded, encoder_mask, key, positions)
        for key, positions in targets.items()
    }


def cells(
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
        return mean, vectors[:, ::-1][:, :SPAN_RANK].T


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
    for row in rows("span moments"):
        predicted = predictions(inputs(row), SPAN_TARGETS)
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
        codes, kept, tails = cells(predicted[image_key])
        values[f"{survey}_codes"] = codes.reshape(-1)
        values[f"{survey}_log_probabilities"] = kept.reshape(-1)
        values[f"{survey}_tails"] = tails
        values[f"{survey}_scalars"] = np.concatenate(
            [predicted[key][0] for key in SCALARS[survey]]
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
