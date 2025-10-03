"""Reproduce the reward misspecification case study for LunarLander."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / ".mpl").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".mpl"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "none")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_HANDLE_SIGNALS", "FALSE")
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [
    {
        "name": "baseline",
        "label": "Baseline reward",
        "env_id": "LunarLander-v3",
        "eval_env_id": "LunarLander-v3",
    },
    {
        "name": "misspecified",
        "label": "Fuel bonus (misspecified)",
        "env_id": "LunarLanderFuelBonus-v0",
        "eval_env_id": "LunarLander-v3",
    },
    {
        "name": "mitigated",
        "label": "Fuel bonus clipped",
        "env_id": "LunarLanderFuelBonusClipped-v0",
        "eval_env_id": "LunarLander-v3",
    },
]

TOTAL_STEPS = 100_000
EVAL_INTERVAL = 10_000
SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward misspecification reproduction suite")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip RL training and emit synthetic data (useful when compute is constrained)",
    )
    return parser.parse_args()


def run_training_suite() -> List[Dict[str, float]]:
    from dataclasses import asdict

    import torch

    from rl_algs.agents import PPOAgent, PPOConfig
    from rl_algs.training import Trainer, TrainerConfig
    from rl_algs.utils import make_env

    def make_agent(env_id: str) -> PPOAgent:
        env = make_env(env_id, seed=SEED)
        obs_space = env.observation_space
        action_space = env.action_space
        env.close()

        obs_dim = int(np.prod(obs_space.shape))
        action_dim = action_space.n

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        config = PPOConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
            seed=SEED,
            rollout_length=2048,
            minibatch_size=128,
            lr=3e-4,
            num_epochs=4,
        )
        return PPOAgent(config)

    def run_scenario(scenario: Dict[str, str]) -> Dict[str, float]:
        agent = make_agent(scenario["env_id"])
        trainer = Trainer(
            agent,
            TrainerConfig(
                env_id=scenario["env_id"],
                eval_env_id=scenario["eval_env_id"],
                total_steps=TOTAL_STEPS,
                eval_interval=EVAL_INTERVAL,
                eval_episodes=10,
                seed=SEED,
            ),
        )
        history = trainer.fit()

        history_path = RESULTS_DIR / f"{scenario['name']}_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        observed_returns = [entry["return"] for entry in history if "return" in entry]
        true_returns = [entry.get("true_return") for entry in history if "true_return" in entry]
        eval_returns = [entry["eval_return"] for entry in history if "eval_return" in entry]
        eval_steps = [entry["step"] for entry in history if "eval_return" in entry]

        summary = {
            "scenario": scenario["label"],
            "env_id": scenario["env_id"],
            "eval_env_id": scenario["eval_env_id"],
            "final_eval_return": float(eval_returns[-1]) if eval_returns else float("nan"),
            "mean_eval_return": float(np.mean(eval_returns[-3:])) if len(eval_returns) >= 3 else float("nan"),
            "mean_observed_return": float(np.mean(observed_returns[-10:])) if observed_returns else float("nan"),
            "mean_true_return": float(np.mean(true_returns[-10:])) if true_returns else float("nan"),
            "eval_steps": eval_steps,
            "eval_curve": eval_returns,
            "history_path": str(history_path.relative_to(PROJECT_ROOT)),
            "agent_config": asdict(agent.config),
        }
        return summary

    return [run_scenario(s) for s in SCENARIOS]


def synthetic_histories() -> List[Dict[str, float]]:
    steps = list(range(EVAL_INTERVAL, TOTAL_STEPS + EVAL_INTERVAL, EVAL_INTERVAL))
    synthetic_curves = {
        "baseline": {
            "observed": [30, 80, 130, 170, 200, 220, 230, 235, 238, 240],
            "true": [30, 80, 130, 170, 200, 220, 230, 235, 238, 240],
            "eval": [30, 80, 130, 170, 200, 220, 230, 235, 238, 240],
        },
        "misspecified": {
            "observed": [60, 120, 180, 240, 280, 320, 340, 355, 365, 370],
            "true": [-110, -90, -70, -50, -20, -25, -45, -70, -80, -85],
            "eval": [-110, -90, -70, -50, -20, -25, -45, -70, -80, -85],
        },
        "mitigated": {
            "observed": [35, 90, 140, 175, 195, 210, 220, 226, 229, 230],
            "true": [35, 85, 135, 165, 185, 205, 215, 222, 226, 229],
            "eval": [35, 85, 135, 165, 185, 205, 215, 222, 226, 229],
        },
    }

    summaries: List[Dict[str, float]] = []
    for scenario in SCENARIOS:
        curves = synthetic_curves[scenario["name"]]
        history: List[Dict[str, float]] = []
        for step, observed, true_val in zip(steps, curves["observed"], curves["true"]):
            entry = {"step": float(step), "return": float(observed), "true_return": float(true_val)}
            history.append(entry)
        for step, eval_val in zip(steps, curves["eval"]):
            history.append({"step": float(step), "eval_return": float(eval_val)})
        history_path = RESULTS_DIR / f"{scenario['name']}_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        summary = {
            "scenario": scenario["label"],
            "env_id": scenario["env_id"],
            "eval_env_id": scenario["eval_env_id"],
            "final_eval_return": float(curves["eval"][-1]),
            "mean_eval_return": float(np.mean(curves["eval"][-3:])),
            "mean_observed_return": float(np.mean(curves["observed"][-3:])),
            "mean_true_return": float(np.mean(curves["true"][-3:])),
            "eval_steps": steps,
            "eval_curve": curves["eval"],
            "history_path": str(history_path.relative_to(PROJECT_ROOT)),
            "agent_config": {},
        }
        summaries.append(summary)
    return summaries


def plot_results(summaries: List[Dict[str, float]]) -> Path:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for summary in summaries:
        ax.plot(summary["eval_steps"], summary["eval_curve"], label=summary["scenario"])
    ax.set_title("Evaluation return with true reward")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Return")
    ax.legend()

    ax2 = axes[1]
    history_path = RESULTS_DIR / "misspecified_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        observed = [(entry["step"], entry["return"]) for entry in history if "return" in entry]
        true_vals = [(entry["step"], entry["true_return"]) for entry in history if "true_return" in entry]
        if observed:
            steps_obs, returns = zip(*observed)
            ax2.plot(steps_obs, returns, label="Observed (misspecified)", alpha=0.8)
        if true_vals:
            steps_true, true_r = zip(*true_vals)
            ax2.plot(steps_true, true_r, label="True reward", alpha=0.8)
        ax2.set_title("Misspecified reward vs. ground truth")
        ax2.set_xlabel("Environment step")
        ax2.set_ylabel("Return")
        ax2.legend()
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "lunar_lander_reward_misspec.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def write_summary_csv(summaries: List[Dict[str, float]]) -> Path:
    summary_path = RESULTS_DIR / "reward_misspec_summary.csv"
    fieldnames = [
        "scenario",
        "env_id",
        "eval_env_id",
        "final_eval_return",
        "mean_eval_return",
        "mean_observed_return",
        "mean_true_return",
        "history_path",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fieldnames})
    return summary_path


def main() -> None:
    args = parse_args()
    if args.synthetic:
        summaries = synthetic_histories()
    else:
        summaries = run_training_suite()

    summary_path = write_summary_csv(summaries)
    plot_path = plot_results(summaries)

    print("Wrote summary to", summary_path)
    print("Saved figure to", plot_path)


if __name__ == "__main__":
    main()
