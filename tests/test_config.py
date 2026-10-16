import pyarrow.parquet as pq

from app.config import N_MORPHOLOGIES, artifact, labels


def test_labels_count_each_morphology_with_unlabelled_galaxies_last(tree) -> None:
    stored = pq.read_table(artifact("mean_points"), columns=["category"])[
        "category"
    ].to_pylist()

    assert labels() == [stored.count(value) for value in [*range(N_MORPHOLOGIES), None]]
