"""Proximal Policy Optimization agent."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from rl_algs.agents.base import Agent, AgentConfig
from rl_algs.networks import MLP


@dataclass
class PPOConfig(AgentConfig):
    obs_dim: int = 0
    action_dim: int = 0
    hidden_dims: tuple[int, ...] = (64, 64)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    rollout_length: int = 2_048
    minibatch_size: int = 64
    num_epochs: int = 4


class RolloutBuffer:
    """Simple list-based rollout storage."""

    def __init__(self) -> None:
        self.clear()

    def add(self, data: Mapping[str, torch.Tensor]) -> None:
        for key, value in data.items():
            self.storage[key].append(value)

    def is_ready(self, length: int) -> bool:
        return len(self.storage["obs"]) >= length

    def get(self) -> Dict[str, torch.Tensor]:
        return {key: torch.cat(values, dim=0) for key, values in self.storage.items()}

    def clear(self) -> None:
        self.storage: Dict[str, List[torch.Tensor]] = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "next_obs": [],
        }


class PPOAgent(Agent):
    """Discrete-action PPO agent."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)
        if config.obs_dim <= 0 or config.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.config = config
        self.actor = MLP(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.critic = MLP(config.obs_dim, 1, config.hidden_dims).to(self.device)
        self._parameters = list(itertools.chain(self.actor.parameters(), self.critic.parameters()))
        self.optimizer = torch.optim.Adam(self._parameters, lr=config.lr)
        self.buffer = RolloutBuffer()

    def select_action(self, obs: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        obs = obs.to(self.device)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if evaluation:
            return torch.argmax(logits, dim=-1)
        action = dist.sample()
        return action

    def store_transition(self, transition: Mapping[str, torch.Tensor]) -> None:  # type: ignore[override]
        required = {"obs", "action", "reward", "next_obs", "done"}
        if not required.issubset(transition):
            raise KeyError(f"Transition must include keys: {required}")

        obs = transition["obs"].to(self.device).view(1, -1)
        action = transition["action"].to(self.device).view(1, -1).long()
        reward = transition["reward"].to(self.device).view(1, 1)
        next_obs = transition["next_obs"].to(self.device).view(1, -1)
        done = transition["done"].to(self.device).view(1, 1)

        with torch.no_grad():
            dist = Categorical(logits=self.actor(obs))
            log_prob = dist.log_prob(action.squeeze(-1)).view(1, 1)
            value = self.critic(obs)

        data = {
            "obs": obs.detach().cpu(),
            "actions": action.detach().cpu(),
            "log_probs": log_prob.detach().cpu(),
            "rewards": reward.detach().cpu(),
            "dones": done.detach().cpu(),
            "values": value.detach().cpu(),
            "next_obs": next_obs.detach().cpu(),
        }
        self.buffer.add(data)

    def update(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        if not batch:
            raise ValueError("PPO update requires a batch; call collect_rollout first")

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device).long().squeeze(-1)
        old_log_probs = batch["log_probs"].to(self.device).squeeze(-1)
        rewards = batch["rewards"].to(self.device).squeeze(-1)
        dones = batch["dones"].to(self.device).squeeze(-1)
        values = batch["values"].to(self.device).squeeze(-1)
        next_obs = batch["next_obs"].to(self.device)

        with torch.no_grad():
            next_values = self.critic(next_obs).squeeze(-1)

        advantages_list = []
        gae = torch.tensor(0.0, device=self.device)
        for step in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_values[step] * mask - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages_list.insert(0, gae)
        advantages = torch.stack(advantages_list)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        total_entropy = 0.0
        total_value = 0.0
        total_policy = 0.0

        batch_size = obs.shape[0]
        for _ in range(self.config.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_idx = indices[start:end]
                if mb_idx.numel() == 0:
                    continue
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]

                dist = Categorical(logits=self.actor(mb_obs))
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (self.critic(mb_obs).squeeze(-1) - mb_returns).pow(2).mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._parameters, max_norm=0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_entropy += float(entropy.item())
                total_value += float(value_loss.item())
                total_policy += float(policy_loss.item())

        num_minibatches = max(1, (batch_size + self.config.minibatch_size - 1) // self.config.minibatch_size)
        num_updates = self.config.num_epochs * num_minibatches
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy / num_updates,
            "value_loss": total_value / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def sample_and_update(self) -> Mapping[str, float]:
        if not self.buffer.is_ready(self.config.rollout_length):
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        batch = self.buffer.get()
        metrics = self.update(batch)
        self.buffer.clear()
        return metrics

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return {
            "config": self.config,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.buffer.clear()
