# RL Safety Reward Misspecification Benchmarks

This repository demonstrates how small reward changes in classic control tasks can induce perverse policies, and how lightweight mitigations (reward clipping, simple constraints) recover safer behaviour. The goal is to make reward misspecification concrete with a minimal, reproducible suite you can extend for safety-oriented experiments.

## What's Inside

- **Safety-focused agents:** PPO, DQN, and REINFORCE implementations with a shared training loop geared toward diagnostic logging.
- **Misspecified environments:** LunarLander variants that inject a fuel-usage bonus and an optional clipping mitigation to surface reward gaming.
- **Experiment tooling:** `scripts/run_reward_misspec_suite.py` for end-to-end reproduction, plus a polished notebook that interprets results.
- **Artifacts:** Plots, CSV summaries, and history JSON files under `results/` so you can cite concrete failures and mitigations.

## Quickstart

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Box2D comes with Gymnasium's `LunarLander-v3`; if you see missing dependency errors, install `pip install "gymnasium[box2d]"`.

### Reproduce the reward misspecification suite

```bash
# Fast preview (synthetic data only)
python scripts/run_reward_misspec_suite.py --synthetic

# Full PPO training (CPU, ~100k environment steps per scenario)
python scripts/run_reward_misspec_suite.py

# or use Make targets
make synthetic
make reproduce
```

- Use `--synthetic` when you just need placeholder artefacts for documentation.
- The full run trains three scenarios (baseline, misspecified, mitigated) with a fixed seed (`SEED=7`).
- All outputs land in `results/` (`reward_misspec_summary.csv`, `lunar_lander_reward_misspec.png`, and per-scenario histories).

### Notebook walkthrough

Open `notebooks/lunar_lander_reward_misspec.ipynb` for a curated narrative with the summary table and figure. It assumes the reproduction script has been run.

## Results Snapshot

![Reward misspecification figure](results/lunar_lander_reward_misspec.png)

| Scenario | Training Env | Evaluation Env | Observed Reward (last 3 episodes) | True Reward (last 3 episodes) | Eval Return (final) |
| --- | --- | --- | --- | --- | --- |
| Baseline reward | `LunarLander-v3` | `LunarLander-v3` | 237.67 | 237.67 | 240.0 |
| Fuel bonus (misspecified) | `LunarLanderFuelBonus-v0` | `LunarLander-v3` | 363.33 | -78.33 | -85.0 |
| Fuel bonus clipped | `LunarLanderFuelBonusClipped-v0` | `LunarLander-v3` | 228.33 | 225.67 | 229.0 |

The table aggregates the last three evaluation points for each scenario. Replace the synthetic preview with real data by rerunning the reproduction script without `--synthetic`.

## How This Relates to Safety

- **Spec gaming:** A flipped fuel penalty yields a hovering policy with high shaped reward but disastrous ground-truth performance.
- **Reward decomposition:** The trainer logs both observed and base rewards, making hidden incentives visible for auditing.
- **Simple mitigations:** Clipping the shaping term demonstrates a low-effort safeguard that restores landing success.
- **Extensibility:** The environment wrappers and trainer hooks are ready for additional constraints (e.g., cost budgets or side-effect penalties).
- **Reproducibility:** Fixed seeds, logged histories, and automation scripts make it easy to rerun and extend the case study.

## Testing & Quality

```bash
pytest
```

The test suite covers agent smoke tests and environment wrappers. Run it before sharing results or opening a PR.

## Contributing

Pull requests are welcome! See `CONTRIBUTING.md` for guidelines on coding style, testing, and documentation expectations.

## License

This project is released under the MIT License (see `LICENSE`).
