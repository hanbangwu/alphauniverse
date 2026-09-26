from functools import cache

from datasets import Dataset, load_dataset


@cache
def dataset(dataset_id: str, dataset_revision: str) -> Dataset:
    return load_dataset(dataset_id, split="train", revision=dataset_revision)
