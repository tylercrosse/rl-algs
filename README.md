# RL Algorithms Reference

A PyTorch-based playground for classic reinforcement learning algorithms. The goal of this
repository is to provide approachable yet extensible implementations that you can use to study
algorithmic differences, run small experiments, and demonstrate familiarity with modern RL tools.

## What's Included

- **Agents:** Deep Q-Network (DQN), REINFORCE, and Proximal Policy Optimization (PPO) implementations
  with shared abstractions.
- **Training Loop:** A generic trainer that handles interaction with Gymnasium environments, logging,
  and evaluation.
- **Utilities:** Common neural network modules, replay buffers, environment helpers, and configuration
  examples for CartPole.
- **Experiment Script:** `scripts/run_experiment.py` wires everything together for quick experiments.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Train an agent:

```bash
python scripts/run_experiment.py --algo dqn --env-id CartPole-v1 --total-steps 50000
```

Swap `--algo` between `dqn`, `ppo`, and `reinforce` to compare behaviors. The trainer logs training
metrics and runs periodic evaluation episodes.

## Repository Structure

```
rl_algs/
  agents/           # Algorithm implementations and configs
  data/             # Experience replay utilities
  networks/         # Shared neural network modules
  training/         # Trainer + evaluator helpers
  utils/            # Environment + torch helpers
configs/            # YAML examples with hyperparameters
scripts/            # Experiment entry points
tests/              # Sanity checks for core components
```

## Next Steps

- Extend the trainer to log metrics via TensorBoard or Weights & Biases.
- Add support for continuous control (e.g., DDPG, SAC) and vectorized environments.
- Introduce Hydra or Pydantic-based configuration management to load the YAML examples directly.
- Create notebooks summarizing comparative results across algorithms and environments.

## Testing

Run the lightweight unit tests:

```bash
pytest
```

These tests focus on smoke-testing the agent APIs. Add environment-specific regression tests as you
build out experiments.
