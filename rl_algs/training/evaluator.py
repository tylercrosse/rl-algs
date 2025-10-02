"""Utilities for evaluating agents without gradient tracking."""

from __future__ import annotations

import numpy as np
import torch

from rl_algs.agents.base import Agent
from rl_algs.utils import to_tensor


def evaluate_agent(agent: Agent, env, episodes: int = 5) -> float:
    """Run evaluation episodes and return the mean return."""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = to_tensor(obs, agent.device).unsqueeze(0)
            with torch.no_grad():
                action_tensor = agent.select_action(obs_tensor, evaluation=True)
            action_np = action_tensor.squeeze(0).detach().cpu().numpy()
            action_value = int(action_np.item()) if action_np.size == 1 else action_np
            obs, reward, terminated, truncated, _ = env.step(action_value)
            obs = np.asarray(obs, dtype=np.float32)
            done = bool(terminated or truncated)
            total_reward += reward
        returns.append(total_reward)
    return float(np.mean(returns)) if returns else 0.0
