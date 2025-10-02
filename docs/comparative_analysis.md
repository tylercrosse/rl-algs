# Comparative Analysis Playbook

This document sketches a lightweight process for running and analyzing experiments across the
implemented algorithms. Treat it as a starting point and adapt as you layer on new agents or tasks.

## 1. Define Questions

- Sample efficiency on classic control: How quickly do DQN, PPO, and REINFORCE solve CartPole-v1?
- Stability: Which algorithms exhibit lower variance in episodic returns across seeds?
- Hyperparameter sensitivity: Which knobs (learning rate, epsilon, clip range) matter most per agent?

## 2. Experimental Setup

1. Choose evaluation environments (e.g., CartPole-v1, LunarLander-v2, Acrobot-v1).
2. Reuse `scripts/run_experiment.py` for initial sweeps; wire in a logging backend (TensorBoard or
   Weights & Biases) to capture metrics in a structured format.
3. Run `N` seeds per configuration (recommend `N >= 5` for meaningful variance estimates).

## 3. Metrics to Collect

- Episodic return (mean ± std) over training steps.
- Evaluation return at fixed intervals (already supported in the trainer).
- Loss curves (value, policy, TD error) to diagnose instabilities.
- Wall-clock time per algorithm for a fixed number of steps.

## 4. Suggested Workflow

1. Extend the trainer to emit metrics to a logging sink. Minimal option: buffer `history` to disk as
   JSON/CSV for later plotting.
2. Create configuration files per algorithm/environment and automate sweeps using a shell script or
   Hydra to manage combinations.
3. After runs finish, aggregate results with a Jupyter notebook (e.g., pandas + seaborn) and generate
   comparison plots.
4. Summarize findings in a short report or README section, highlighting trade-offs and surprising
   behaviors.

## 5. Portfolio-Ready Presentation

- Maintain clean experiment logs and plots under a `reports/` directory.
- Add Markdown summaries that connect observed behavior back to algorithm design choices.
- Include reproducibility instructions (seed values, command invocations, dependency versions).

## 6. Future Extensions

- Continuous control benchmarks (Pendulum, MountainCarContinuous) with actor-critic methods.
- Offline RL baselines using datasets from `D4RL`.
- Integrate unit tests for environment wrappers and numerical checks (e.g., value targets).
- Deploy CI to run smoke tests on pull requests.
