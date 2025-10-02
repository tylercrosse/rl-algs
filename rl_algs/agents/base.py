"""Base interfaces that concrete agents build upon."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch


class TrainState(Protocol):
    """Protocol for the dict-like training state stored by agents."""

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


@dataclass
class AgentConfig:
    """Base configuration shared across algorithms."""

    gamma: float = 0.99
    device: str | torch.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed: int = 0


class Agent(abc.ABC):
    """Abstract agent API for training and inference."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")
        torch.manual_seed(config.seed)

    @abc.abstractmethod
    def select_action(self, obs: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        """Return an action for the given observation."""

    def store_transition(self, transition: Mapping[str, torch.Tensor]) -> None:
        """Optional hook for agents that require replay buffers."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        """Perform a single optimization step and return logging metrics."""

    def to(self, device: str | torch.device) -> None:
        """Record the device used by the agent and update random seeds."""
        self.device = torch.device(device)
        self.config.device = self.device
        torch.manual_seed(self.config.seed)

    def state_dict(self) -> Mapping[str, Any]:
        """Return the state that should be checkpointed."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state from a checkpoint."""
        raise NotImplementedError
