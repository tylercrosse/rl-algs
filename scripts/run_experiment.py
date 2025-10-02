"""CLI entry point for training RL agents."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Mapping

import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    parser.add_argument("--plot", action="store_true", help="Generate a timestep vs. return plot after training")
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Where to write the plot when --plot is enabled (defaults to training_returns.png)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Window size for rolling-average returns in the plot",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="rl-algs", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity or team name")
    parser.add_argument("--wandb-name", default=None, help="Custom run name registered in W&B")
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        help="Tag to attach to the W&B run (can be passed multiple times)",
    )
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
    loggers = []
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Weights & Biases is required for --wandb; install it via `pip install .[logging]`"
            ) from exc

        wandb_config = _prepare_wandb_config(args, trainer_config, agent, obs_dim, action_dim)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tag or None,
            config=wandb_config,
        )

        def _wandb_logger(metrics: Mapping[str, float]) -> None:
            wandb_run.log(dict(metrics))

        loggers.append(_wandb_logger)

    trainer = Trainer(agent, trainer_config, loggers=loggers)
    history: List[Dict[str, float]] = []
    try:
        history = trainer.fit()

        final_eval = [entry["eval_return"] for entry in history if "eval_return" in entry]
        if final_eval:
            print(f"Final evaluation return: {final_eval[-1]:.2f}")
            if wandb_run is not None:
                wandb_run.summary["final_eval_return"] = final_eval[-1]

        if args.plot or args.plot_path:
            _generate_plot(
                args,
                history,
                title=f"{args.algo.upper()} on {args.env_id}",
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _plot_history(history: List[Dict[str, float]], path: str, title: str, rolling_window: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plotting; install it via `pip install matplotlib`"
        ) from exc

    episode_points = [(entry["step"], entry["return"]) for entry in history if "return" in entry]
    eval_points = [(entry["step"], entry["eval_return"]) for entry in history if "eval_return" in entry]

    if not episode_points and not eval_points:
        print("No return metrics captured; skipping plot generation")
        return

    plt.figure(figsize=(8, 5))
    if episode_points:
        episode_points.sort()
        steps, returns = zip(*episode_points)
        steps = np.asarray(steps)
        returns = np.asarray(returns, dtype=np.float32)
        if len(returns) >= rolling_window:
            kernel = np.ones(rolling_window, dtype=np.float32) / rolling_window
            smoothed = np.convolve(returns, kernel, mode="valid")
            smoothed_steps = steps[rolling_window - 1 :]
            plt.plot(steps, returns, label="Episode return", alpha=0.3)
            plt.plot(smoothed_steps, smoothed, label=f"Episode return (rolling {rolling_window})", alpha=0.9)
        else:
            plt.plot(steps, returns, label="Episode return", alpha=0.8)
    if eval_points:
        eval_steps, eval_returns = zip(*eval_points)
        plt.plot(eval_steps, eval_returns, label="Eval return", linestyle="--", alpha=0.8)

    plt.xlabel("Environment step")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved plot to {path}")


def _generate_plot(args: argparse.Namespace, history: List[Dict[str, float]], title: str) -> None:
    # Ensure results directory exists in the project root and prefer it for plots.
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    if args.plot_path:
        plot_path = args.plot_path
        # If the user provided only a filename (no directory) and it's not absolute,
        # place it inside the results/ directory.
        if not os.path.isabs(plot_path) and not os.path.dirname(plot_path):
            plot_path = os.path.join(results_dir, plot_path)
    else:
        # sanitize env id and algo for filename
        safe_env = args.env_id.replace("/", "-").replace("\\", "-")
        safe_algo = args.algo.replace("/", "-").replace("\\", "-")
        plot_filename = f"{safe_algo}_{safe_env}_{args.total_steps}.png"
        plot_path = os.path.join(results_dir, plot_filename)

    _plot_history(
        history,
        plot_path,
        title=title,
        rolling_window=max(1, args.rolling_window),
    )


def _prepare_wandb_config(
    args: argparse.Namespace,
    trainer_config: TrainerConfig,
    agent,
    obs_dim: int,
    action_dim: int,
) -> Dict[str, object]:
    agent_config = asdict(agent.config)
    trainer_cfg = asdict(trainer_config)
    return {
        "algo": args.algo,
        "env_id": args.env_id,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "agent": agent_config,
        "trainer": trainer_cfg,
    }


if __name__ == "__main__":
    main()
