import importlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from app.config import FLAG_SURVEYS, POINTS, artifact, galaxy_count
from scripts.fixture import build, forget

pytest.importorskip("torch")
umap_module = importlib.import_module("app.parametric_umap")

LABELS = [1, None, 3, 0]
PROJECTIONS = ("mean_points", "full_points")


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[list[dict[str, pa.Table]]]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path_factory.mktemp("umap")))
        patch.setattr(umap_module, "WANDB_MODE", "disabled")
        patch.setattr(umap_module, "EPOCHS", 1)
        patch.setattr(
            umap_module,
            "dataset",
            lambda *_: Dataset.from_dict({FLAG_SURVEYS["gz10"]: LABELS}),
        )
        try:
            build(len(LABELS))
            tables = []
            for _ in range(2):
                for role in PROJECTIONS:
                    artifact(role).unlink()
                umap_module.generate_projections()
                tables.append(
                    {role: pq.read_table(artifact(role)) for role in PROJECTIONS}
                )
            yield tables
        finally:
            forget()


def test_projections_have_the_points_schema(runs: list[dict[str, pa.Table]]) -> None:
    for table in runs[0].values():
        assert table.schema.equals(POINTS)


def test_each_galaxy_gets_one_mean_point_with_its_label(
    runs: list[dict[str, pa.Table]],
) -> None:
    mean_points = runs[0]["mean_points"]

    assert mean_points.column("galaxy").to_pylist() == list(range(len(LABELS)))
    assert mean_points.column("category").to_pylist() == LABELS


def test_projections_are_deterministic(runs: list[dict[str, pa.Table]]) -> None:
    first, second = runs

    for role in first:
        assert first[role].equals(second[role])


def test_scan_returns_each_chosen_embedding_in_the_order_asked(tree: Path) -> None:
    streamed = np.concatenate([values for _, _, values in umap_module._stream()])
    chosen = np.random.default_rng(0).permutation(len(streamed))[: len(streamed) // 3]

    _, taken = umap_module.scan(galaxy_count(), chosen)

    np.testing.assert_array_equal(taken, streamed[chosen])


def test_scan_rejects_a_sample_past_the_streamed_embeddings(tree: Path) -> None:
    with pytest.raises(ValueError, match="streamed embeddings"):
        umap_module.scan(galaxy_count(), np.array([umap_module.embedding_count()]))
