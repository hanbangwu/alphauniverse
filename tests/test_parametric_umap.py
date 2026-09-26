from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from app import search as search_module
from app.config import FLAG_SURVEYS, POINTS, artifact
from scripts.fixture import build

torch = pytest.importorskip("torch")
umap_module = pytest.importorskip("app.parametric_umap")

LABELS = [1, None, 3, 0]
CACHES = (
    search_module.source,
    search_module.index,
    search_module.with_spectrum,
    search_module.starts,
)


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[list[dict[str, pa.Table]]]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path_factory.mktemp("umap")))
        patch.setattr(umap_module, "WANDB_MODE", "disabled")
        patch.setattr(
            umap_module,
            "dataset",
            lambda *_: Dataset.from_dict({FLAG_SURVEYS["gz10"]: LABELS}),
        )
        for cache in CACHES:
            cache.cache_clear()
        try:
            build(len(LABELS))
            for cache in CACHES:
                cache.cache_clear()
            tables = []
            for _ in range(2):
                umap_module.generate_projections()
                tables.append(
                    {
                        role: pq.read_table(artifact(role))
                        for role in ("mean_points", "full_points")
                    }
                )
            yield tables
        finally:
            for cache in CACHES:
                cache.cache_clear()


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
