"""Small helpers for PyTorch interoperability."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def to_tensor(array: np.ndarray | Iterable[float] | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Convert an array-like input to a float32 tensor on the given device."""
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.as_tensor(array, dtype=torch.float32, device=device)
