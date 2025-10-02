"""REINFORCE (vanilla policy gradient) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from rl_algs.agents.base import Agent, AgentConfig
from rl_algs.networks import MLP


@dataclass
class ReinforceConfig(AgentConfig):
    obs_dim: int = 0
    action_dim: int = 0
    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01


class ReinforceAgent(Agent):
    """Classic on-policy policy gradient agent."""

    def __init__(self, config: ReinforceConfig) -> None:
        super().__init__(config)
        if config.obs_dim <= 0 or config.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.config = config
        self.policy = MLP(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[torch.Tensor] = []
        self._entropies: List[torch.Tensor] = []

    def select_action(self, obs: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        logits = self.policy(obs)
        if evaluation:
            return torch.argmax(logits, dim=-1)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action).reshape(-1))
        self._entropies.append(dist.entropy().reshape(-1))
        return action

    def record_reward(self, reward: float) -> None:
        self._rewards.append(torch.tensor([reward], device=self.device))

    def end_episode(self) -> Mapping[str, float]:
        if not self._rewards:
            return {"loss": 0.0, "return": 0.0}

        returns = []
        g = torch.zeros(1, device=self.device)
        for reward in reversed(self._rewards):
            g = reward + self.config.gamma * g
            returns.append(g)
        returns = torch.cat(list(reversed(returns)))
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs = torch.cat(self._log_probs)
        entropies = torch.cat(self._entropies)

        loss = -(log_probs * returns.detach()).mean()
        if self.config.entropy_coef:
            loss -= self.config.entropy_coef * entropies.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()

        episode_return = float(torch.sum(torch.cat(self._rewards)).item())
        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()
        return {"loss": float(loss.item()), "return": episode_return}

    def update(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        raise NotImplementedError("Use end_episode() to trigger optimization for REINFORCE")

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return {
            "config": self.config,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
