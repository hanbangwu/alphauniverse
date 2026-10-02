from functools import cache

from datasets import Dataset, load_dataset

from .config import DATASET_ID, DATASET_REVISION


@cache
def dataset() -> Dataset:
    return load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
