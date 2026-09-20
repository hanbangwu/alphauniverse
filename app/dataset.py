"""Access to the source Hugging Face dataset.

Build-time only. Nothing served imports this: cutouts are precomputed by
`app/cutouts.py` and embeddings by `app/encode.py`.
"""

from functools import cache

from datasets import Dataset, load_dataset


@cache
def dataset(dataset_id: str, dataset_revision: str) -> Dataset:
    """The dataset's train split, downloaded and memory-mapped once per process."""
    return load_dataset(dataset_id, split="train", revision=dataset_revision)
