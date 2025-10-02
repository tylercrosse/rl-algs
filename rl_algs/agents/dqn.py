"""Deep Q-Network agent implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn

from rl_algs.agents.base import Agent, AgentConfig
from rl_algs.data.replay_buffer import ReplayBuffer, ReplayBufferConfig
from rl_algs.networks import MLP


@dataclass
class DQNConfig(AgentConfig):
    obs_dim: int = 0
    action_dim: int = 0
    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    target_update_interval: int = 100
    min_buffer_size: int = 1_000
    replay_buffer: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig(capacity=100_000, batch_size=64))
    double_q: bool = True
    gradient_clip_norm: float | None = 10.0
    exploration_epsilon: float = 0.1
    evaluation_epsilon: float = 0.0


class DQNAgent(Agent):
    """DQN with target network and optional Double Q-learning."""

    def __init__(self, config: DQNConfig) -> None:
        super().__init__(config)
        if config.obs_dim <= 0 or config.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.config = config
        self.q_network = MLP(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.target_network = MLP(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.obs_dim, config.action_dim, config.replay_buffer)
        self.loss_fn = nn.SmoothL1Loss()
        self._steps = 0

    def select_action(self, obs: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        eps = self.config.evaluation_epsilon if evaluation else self.config.exploration_epsilon
        if torch.rand(1).item() < eps:
            return torch.randint(0, self.config.action_dim, (obs.shape[0],), device=obs.device)

        with torch.no_grad():
            q_values = self.q_network(obs)
            return torch.argmax(q_values, dim=-1)

    def store_transition(self, transition: Mapping[str, torch.Tensor]) -> None:  # type: ignore[override]
        if not {"obs", "action", "reward", "next_obs", "done"}.issubset(transition):
            raise KeyError("Transition must include obs, action, reward, next_obs, and done")
        self.replay_buffer.add(
            transition["obs"].cpu().numpy(),
            transition["action"].cpu().numpy(),
            float(transition["reward"].item()),
            transition["next_obs"].cpu().numpy(),
            bool(transition["done"].item()),
        )

    def update(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:  # type: ignore[override]
        self._steps += 1
        obs = batch["obs"].to(self.device)
        actions = batch["action"].long().to(self.device)
        rewards = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["done"].to(self.device)

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.config.double_q:
                next_actions = torch.argmax(self.q_network(next_obs), dim=-1, keepdim=True)
                next_q = self.target_network(next_obs).gather(1, next_actions).squeeze(-1)
            else:
                next_q = self.target_network(next_obs).max(dim=-1).values
            targets = rewards.squeeze(-1) + self.config.gamma * (1 - dones.squeeze(-1)) * next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        if self._steps % self.config.target_update_interval == 0:
            self._soft_update()

        return {"loss": loss.item(), "q_mean": q_values.mean().item()}

    def sample_and_update(self) -> Mapping[str, float]:
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {"loss": 0.0, "q_mean": 0.0}
        batch = self.replay_buffer.sample(self.device)
        return self.update(batch)

    def _soft_update(self) -> None:
        tau = self.config.tau
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def to(self, device: str | torch.device) -> None:  # type: ignore[override]
        super().to(device)
        device = torch.device(device)
        self.q_network.to(device)
        self.target_network.to(device)

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return {
            "config": self.config,
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self._steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._steps = state_dict.get("steps", 0)
