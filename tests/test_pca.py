from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
from sklearn.decomposition import PCA

from app import pca
from app.config import AION_EMBEDDING_DIM, SEED
from app.search import source


def test_batched_patterns_match_one_pca_over_every_patch(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pca, "GALAXIES_PER_BATCH", 1)
    pca.generate_patterns()

    ls = source("encoded").to_table(columns=["ls"])["ls"]
    kept = pc.list_slice(ls, 0, pca.PATCHES)
    vectors = pc.list_flatten(pc.list_flatten(kept)).to_numpy()
    patches = vectors.reshape(-1, AION_EMBEDDING_DIM).astype(np.float32)
    scores = PCA(n_components=3, random_state=SEED).fit(patches).transform(patches)
    low, high = np.percentile(scores, pca.PERCENTILES, axis=0)
    expected = np.rint(np.clip((scores - low) / (high - low), 0, 1) * 255)

    stored = source("patterns").to_table()["colours"]
    assert np.array_equal(
        np.stack(stored.to_numpy(zero_copy_only=False)).reshape(-1, 3), expected
    )
