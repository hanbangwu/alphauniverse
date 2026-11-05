from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@cache
def device() -> torch.device:
    import torch

    return torch.accelerator.current_accelerator(check_available=True) or torch.device(
        "cpu"
    )
