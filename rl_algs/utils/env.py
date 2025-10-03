"""Environment helper utilities and registration side-effects."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

# Import custom environments for their registration side-effects. The alias underscores silence linters.
try:  # pragma: no cover - import failure only happens in minimal installations
    import rl_algs.envs as _rl_envs  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - package users may exclude optional envs
    _rl_envs = None


def make_env(env_id: str, seed: int = 0, max_episode_steps: int | None = None, **kwargs: Any) -> gym.Env:
    """Create a Gymnasium environment with optional time limit wrapper."""
    env = gym.make(env_id, **kwargs)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
