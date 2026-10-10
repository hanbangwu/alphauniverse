from collections.abc import Iterator
from itertools import batched

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from . import encode
from .config import (
    ANCHOR,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    REDSHIFT,
    SEED,
    STORE_COLUMNS,
    artifact,
    device,
)
from .search import source

CHUNK = 128
TOP_CODES = 64
SPECTRUM_TOKEN_RANK = 256
VOCABULARY = 1024
REDSHIFT_VOCABULARY = 1025
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

PREDICTIONS = pa.schema(
    [
        pa.field("galaxy", pa.int32()),
        pa.field(REDSHIFT, pa.list_(pa.float16(), REDSHIFT_VOCABULARY)),
    ]
    + [
        field
        for survey, keys in TABLE_VALUES.items()
        for field in (
            pa.field(
                f"{survey}_codes", pa.list_(pa.uint16(), N_IMAGE_TOKENS * TOP_CODES)
            ),
            pa.field(
                f"{survey}_log_probabilities",
                pa.list_(pa.float16(), N_IMAGE_TOKENS * TOP_CODES),
            ),
            pa.field(f"{survey}_tails", pa.list_(pa.float32(), N_IMAGE_TOKENS)),
            pa.field(
                f"{survey}_table_values", pa.list_(pa.float16(), len(keys) * VOCABULARY)
            ),
        )
    ]
    + [
        field
        for survey in SPECTRA
        for field in (
            pa.field(
                f"{survey}_coefficients",
                pa.list_(pa.uint8(), N_SPECTRUM_TOKENS * SPECTRUM_TOKEN_RANK),
            ),
            pa.field(f"{survey}_offsets", pa.list_(pa.float32(), N_SPECTRUM_TOKENS)),
            pa.field(f"{survey}_steps", pa.list_(pa.float32(), N_SPECTRUM_TOKENS)),
        )
    ]
)

SPECTRUM_TOKEN_TARGETS = {
    key: np.arange(1, N_SPECTRUM_TOKENS + 1) for key in SPECTRA.values()
}
TARGETS = (
    {key: np.arange(N_IMAGE_TOKENS) for key in IMAGES.values()}
    | {key: np.arange(1) for keys in TABLE_VALUES.values() for key in keys}
    | {REDSHIFT_KEY: np.arange(1)}
    | SPECTRUM_TOKEN_TARGETS
)


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


def batch(records: list[dict[str, np.ndarray]]) -> pa.RecordBatch:
    columns = []
    for field in PREDICTIONS:
        flat = np.concatenate([values[field.name] for values in records])
        columns.append(
            pa.FixedSizeListArray.from_arrays(flat, field.type.list_size)
            if pa.types.is_fixed_size_list(field.type)
            else pa.array(flat)
        )
    return pa.record_batch(columns, schema=PREDICTIONS)


def generate_predictions() -> None:
    fitted = bases()
    np.savez(
        artifact("prediction_basis"),
        **{
            f"{survey}_{name}": value
            for survey, pair in fitted.items()
            for name, value in zip(("mean", "directions"), pair, strict=True)
        },
    )
    with pa.ipc.new_file(artifact("predictions"), PREDICTIONS) as writer:
        for chunk in batched(rows("predict"), BATCH):
            writer.write_batch(
                batch(
                    [
                        record(row["galaxy"], predictions(inputs(row), TARGETS), fitted)
                        for row in chunk
                    ]
                )
            )
