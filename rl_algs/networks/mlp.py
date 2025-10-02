"""Reusable multilayer perceptron building block."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class MLP(nn.Sequential):
    """Simple MLP with configurable hidden sizes and activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        hidden_dims = list(hidden_dims or [])
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        super().__init__(*layers)

    @torch.no_grad()
    def forward_with_layers(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        """Yield activations after each layer for debugging/analysis."""
        for layer in self:
            x = layer(x)
            yield x
