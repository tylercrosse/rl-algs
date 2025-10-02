"""CLI entry point for training RL agents."""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from rl_algs.agents import DQNAgent, DQNConfig, PPOAgent, PPOConfig, ReinforceAgent, ReinforceConfig
from rl_algs.training.trainer import Trainer, TrainerConfig
from rl_algs.utils import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents on a Gymnasium task")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo"], default="dqn", help="Which agent to train")
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--total-steps", type=int, default=50_000, help="Number of environment steps to collect")
    parser.add_argument("--device", default="cpu", help="Torch device to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=10_000, help="Evaluation interval in environment steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    probe_env = make_env(args.env_id, seed=args.seed)
    obs_space = probe_env.observation_space
    action_space = probe_env.action_space

    if len(obs_space.shape) != 1:
        raise ValueError("Current scaffolding assumes a 1D observation space")
    obs_dim = int(np.prod(obs_space.shape))

    if hasattr(action_space, "n"):
        action_dim = action_space.n
    else:
        raise ValueError("Current scaffolding assumes a discrete action space")

    probe_env.close()

    algo_builders = {
        "dqn": lambda: DQNAgent(DQNConfig(obs_dim=obs_dim, action_dim=action_dim, device=args.device, seed=args.seed)),
        "reinforce": lambda: ReinforceAgent(
            ReinforceConfig(obs_dim=obs_dim, action_dim=action_dim, device=args.device, seed=args.seed)
        ),
        "ppo": lambda: PPOAgent(PPOConfig(obs_dim=obs_dim, action_dim=action_dim, device=args.device, seed=args.seed)),
    }

    agent = algo_builders[args.algo]()
    trainer_config = TrainerConfig(
        env_id=args.env_id,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer = Trainer(agent, trainer_config)
    history: List[Dict[str, float]] = trainer.fit()

    final_eval = [entry["eval_return"] for entry in history if "eval_return" in entry]
    if final_eval:
        print(f"Final evaluation return: {final_eval[-1]:.2f}")


if __name__ == "__main__":
    main()
