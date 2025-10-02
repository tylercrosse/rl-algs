"""Training loop for running RL agents in Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm

from rl_algs.agents.base import Agent
from rl_algs.training.evaluator import evaluate_agent
from rl_algs.utils import make_env, to_tensor


@dataclass
class TrainerConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 5
    max_episode_steps: Optional[int] = None
    seed: int = 0


class Trainer:
    """Generic trainer that minimizes boilerplate across agents."""

    def __init__(self, agent: Agent, config: TrainerConfig) -> None:
        self.agent = agent
        self.config = config
        self.env = make_env(config.env_id, seed=config.seed, max_episode_steps=config.max_episode_steps)
        self.eval_env = make_env(config.env_id, seed=config.seed + 1, max_episode_steps=config.max_episode_steps)
        self.device = agent.device

    def fit(self) -> List[Dict[str, float]]:
        obs, _ = self.env.reset(seed=self.config.seed)
        obs = np.asarray(obs, dtype=np.float32)
        history: List[Dict[str, float]] = []

        for step in tqdm(range(1, self.config.total_steps + 1), desc="Training", unit="step"):
            obs_tensor = to_tensor(obs, self.device).unsqueeze(0)
            action_tensor = self.agent.select_action(obs_tensor)
            action_tensor = action_tensor.detach()
            action_np = action_tensor.squeeze(0).cpu().numpy()
            action_value = int(action_np.item()) if action_np.size == 1 else action_np

            next_obs, reward, terminated, truncated, _ = self.env.step(action_value)
            done = bool(terminated or truncated)

            stored_action = action_tensor.view(1, -1).to(device=self.device, dtype=torch.int64)
            transition = {
                "obs": obs_tensor,
                "action": stored_action,
                "reward": torch.tensor([[reward]], dtype=torch.float32, device=self.device),
                "next_obs": to_tensor(next_obs, self.device).unsqueeze(0),
                "done": torch.tensor([[float(done)]], dtype=torch.float32, device=self.device),
            }

            try:
                self.agent.store_transition(transition)  # type: ignore[attr-defined]
            except AttributeError:
                pass
            except NotImplementedError:
                pass

            if hasattr(self.agent, "record_reward"):
                getattr(self.agent, "record_reward")(reward)

            metrics: Mapping[str, float] = {}
            if hasattr(self.agent, "sample_and_update"):
                metrics = getattr(self.agent, "sample_and_update")()
            if metrics:
                history.append({"step": float(step), **{k: float(v) for k, v in metrics.items()}})

            if done:
                if hasattr(self.agent, "end_episode"):
                    episode_metrics = getattr(self.agent, "end_episode")()
                    if episode_metrics:
                        history.append({"step": float(step), **{k: float(v) for k, v in episode_metrics.items()}})
                obs, _ = self.env.reset()
                obs = np.asarray(obs, dtype=np.float32)
            else:
                obs = np.asarray(next_obs, dtype=np.float32)

            if step % self.config.eval_interval == 0:
                eval_score = evaluate_agent(self.agent, self.eval_env, episodes=self.config.eval_episodes)
                history.append({"step": float(step), "eval_return": eval_score})

        # Final evaluation
        final_eval_score = evaluate_agent(self.agent, self.eval_env, episodes=self.config.eval_episodes)
        history.append({"step": float(self.config.total_steps), "eval_return": final_eval_score})

        return history
