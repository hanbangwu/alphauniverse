from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from . import encode
from .config import ANCHOR, DIM, N_PATCHES, N_SPANS, TOKEN_SURVEYS, artifact, device
from .search import source

CHUNK = 128
SEED = 0
TOP_CODES = 64
SPAN_RANK = 256
VOCABULARY = 1024
LEVELS = 255
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

PREDICTIONS = pa.schema(
    [pa.field("galaxy", pa.int32())]
    + [
        field
        for survey, keys in SCALARS.items()
        for field in (
            pa.field(f"{survey}_codes", pa.list_(pa.uint16(), N_PATCHES * TOP_CODES)),
            pa.field(
                f"{survey}_log_probabilities",
                pa.list_(pa.float16(), N_PATCHES * TOP_CODES),
            ),
            pa.field(f"{survey}_tails", pa.list_(pa.float32(), N_PATCHES)),
            pa.field(
                f"{survey}_scalars", pa.list_(pa.float16(), len(keys) * VOCABULARY)
            ),
        )
    ]
    + [
        field
        for survey in SPECTRA
        for field in (
            pa.field(
                f"{survey}_coefficients", pa.list_(pa.uint8(), N_SPANS * SPAN_RANK)
            ),
            pa.field(f"{survey}_offsets", pa.list_(pa.float32(), N_SPANS)),
            pa.field(f"{survey}_steps", pa.list_(pa.float32(), N_SPANS)),
        )
    ]
)

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


def cells(
    log_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = np.argsort(-log_probabilities, axis=1, kind="stable")[:, :TOP_CODES]
    kept = np.take_along_axis(log_probabilities, codes, axis=1).astype(np.float64)
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


def rows() -> Iterator[dict]:
    for batch in source("tokens").to_batches(columns=["galaxy", *TOKEN_SURVEYS]):
        yield from batch.to_pylist()


def bases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    moments = {survey: Moments() for survey in SPECTRA}
    count = pq.read_metadata(artifact("tokens")).num_rows
    for row in tqdm(rows(), total=count, desc="span moments"):
        predicted = predictions(inputs(row), SPAN_TARGETS)
        for survey, key in SPECTRA.items():
            moments[survey].add(np.exp(predicted[key]))
    return {survey: moment.basis() for survey, moment in moments.items()}


def record(
    galaxy: int,
    predicted: dict[str, np.ndarray],
    fitted: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {"galaxy": np.asarray([galaxy], dtype=np.int32)}
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


def batch(records: list[dict[str, np.ndarray]]) -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array(np.concatenate([values[field.name] for values in records]))
            if field.name == "galaxy"
            else pa.FixedSizeListArray.from_arrays(
                pa.array(np.concatenate([values[field.name] for values in records])),
                field.type.list_size,
            )
            for field in PREDICTIONS
        ],
        schema=PREDICTIONS,
    )


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
    count = pq.read_metadata(artifact("tokens")).num_rows
    with pa.ipc.new_file(artifact("predictions"), PREDICTIONS) as writer:
        records = []
        for row in tqdm(rows(), total=count, desc="predict"):
            records.append(
                record(row["galaxy"], predictions(inputs(row), TARGETS), fitted)
            )
            if len(records) == BATCH:
                writer.write_batch(batch(records))
                records.clear()
        if records:
            writer.write_batch(batch(records))
