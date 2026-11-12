from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from aion.modalities import LegacySurveyImage
from sklearn.decomposition import PCA

from .config import AION_EMBEDDING_DIM, SEED, artifact
from .search import source

PATCHES = LegacySurveyImage.num_tokens
SAMPLE = 100_000
GALAXIES_PER_BATCH = 256
PERCENTILES = (2, 98)


def batches() -> Iterator[tuple[int, np.ndarray]]:
    start = 0
    for batch in source("encoded").to_batches(
        columns=["ls"], batch_size=GALAXIES_PER_BATCH
    ):
        if not batch.num_rows:
            continue
        kept = pc.list_slice(batch["ls"], 0, PATCHES)
        vectors = pc.list_flatten(pc.list_flatten(kept)).to_numpy()
        yield start, vectors.reshape(-1, AION_EMBEDDING_DIM).astype(np.float32)
        start += batch.num_rows * PATCHES


def fitted(total: int) -> PCA:
    chosen = np.sort(
        np.random.default_rng(SEED).choice(total, min(SAMPLE, total), replace=False)
    )
    sample = np.empty((len(chosen), AION_EMBEDDING_DIM), np.float32)
    for start, vectors in batches():
        inside = (chosen >= start) & (chosen < start + len(vectors))
        sample[inside] = vectors[chosen[inside] - start]
    return PCA(n_components=3, random_state=SEED).fit(sample)


def generate_patterns() -> None:
    galaxies = source("encoded").to_table(columns=["galaxy"])["galaxy"]
    scores = np.empty((len(galaxies) * PATCHES, 3), np.float32)
    pca = fitted(len(scores))
    for start, vectors in batches():
        scores[start : start + len(vectors)] = pca.transform(vectors)
    low, high = np.percentile(scores, PERCENTILES, axis=0)
    colours = np.rint(np.clip((scores - low) / (high - low), 0, 1) * 255).astype(
        np.uint8
    )
    table = pa.table(
        {
            "galaxy": galaxies.combine_chunks(),
            "colours": pa.FixedSizeListArray.from_arrays(
                colours.reshape(-1), PATCHES * 3
            ),
        }
    )
    with pa.ipc.new_file(artifact("patterns"), table.schema) as writer:
        writer.write_table(table)
