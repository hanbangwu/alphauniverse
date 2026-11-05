import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split

from core.artifacts import artifact
from core.config import TextModality
from core.dataset import dataset
from core.device import device

from .config import SEED

VALIDATION = 0.3


def pairs(modality: TextModality) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    table = pq.read_table(
        artifact(f"pairs_{modality}"), columns=["galaxy", "aion", "wemm"]
    )
    galaxy = np.asarray(table.column("galaxy"))
    aion, wemm = (
        torch.as_tensor(
            np.asarray(table.column(name).combine_chunks().flatten()).reshape(
                len(galaxy), -1
            ),
            device=device(),
        )
        for name in ("aion", "wemm")
    )
    return galaxy, aion, wemm


def held_out(galaxy: np.ndarray) -> torch.Tensor:
    _, validation = train_test_split(
        np.arange(len(dataset())), test_size=VALIDATION, random_state=SEED
    )
    return torch.as_tensor(np.isin(galaxy, validation), device=device())
