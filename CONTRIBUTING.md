# Contributing Guidelines

Thanks for your interest in extending the reward misspecification benchmarks! To keep changes easy to review and reproduce, please follow these guidelines.

## Development Workflow

- Fork the repository and create a feature branch for your work.
- Install dependencies with `pip install -e .[dev]` (and optionally `pip install "gymnasium[box2d]"`).
- Regenerate artefacts with `python scripts/run_reward_misspec_suite.py` before submitting substantial changes.
- Run `pytest` locally; add focused unit tests for new utilities or environment wrappers.

## Style & Tooling

- This project uses `black` and `ruff` (configured in `pyproject.toml`). Run them before opening a PR:
  ```bash
  black .
  ruff check .
  ```
- Keep comments concise and purposeful—prefer docstrings or high-level explanations over line-by-line narration.
- Default to ASCII unless the surrounding file already uses Unicode symbols.

## Pull Requests

- Include a short summary of the failure mode or mitigation you are adding.
- Attach or describe relevant artefacts (plots, CSV summaries, notebook cells) to demonstrate the behaviour.
- Note any deviations from the default reproduction script and provide command lines for reviewers.

By contributing, you agree that your submissions will be licensed under the MIT License.
