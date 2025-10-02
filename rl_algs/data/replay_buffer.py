"""Replay buffer implementations for experience replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Mapping

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000
    batch_size: int = 64


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, config: ReplayBufferConfig | None = None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or ReplayBufferConfig()
        self._storage: Deque[Mapping[str, np.ndarray]] = deque(maxlen=self.config.capacity)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._storage.append(
            {
                "obs": obs,
                "action": action,
                "reward": np.array([reward], dtype=np.float32),
                "next_obs": next_obs,
                "done": np.array([done], dtype=np.float32),
            }
        )

    def __len__(self) -> int:
        return len(self._storage)

    def sample(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self) < self.config.batch_size:
            raise ValueError("Not enough samples to draw a batch")

        indices = np.random.choice(len(self._storage), self.config.batch_size, replace=False)
        batch = [self._storage[idx] for idx in indices]
        stacked = {
            key: np.stack([entry[key] for entry in batch]) for key in batch[0]
        }
        return {key: torch.from_numpy(value).to(device=device) for key, value in stacked.items()}
