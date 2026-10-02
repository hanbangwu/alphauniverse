from pathlib import Path

import pyarrow.parquet as pq

from app import main
from app.config import artifact


def test_labels_partition_every_galaxy(tree: Path) -> None:
    morphologies, unlabelled = main.labels()

    assert (
        sum(morphologies) + unlabelled
        == pq.read_metadata(artifact("mean_points")).num_rows
    )
