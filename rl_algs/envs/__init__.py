"""Environment registrations for the RL safety micro-benchmarks."""

from __future__ import annotations

from gymnasium.envs.registration import register

from .misspecified_lunar_lander import (
    FuelBonusConfig,
    FuelBonusRewardWrapper,
    make_lunar_lander_fuel_bonus_clipped_env,
    make_lunar_lander_fuel_bonus_env,
)

__all__ = [
    "FuelBonusConfig",
    "FuelBonusRewardWrapper",
    "make_lunar_lander_fuel_bonus_env",
    "make_lunar_lander_fuel_bonus_clipped_env",
]


def _register_envs() -> None:
    """Ensure custom Gym IDs exist; ignore if they are already registered."""
    try:
        register(
            id="LunarLanderFuelBonus-v0",
            entry_point="rl_algs.envs.misspecified_lunar_lander:make_lunar_lander_fuel_bonus_env",
        )
    except Exception:  # pragma: no cover - id may already exist in tests
        pass

    try:
        register(
            id="LunarLanderFuelBonusClipped-v0",
            entry_point="rl_algs.envs.misspecified_lunar_lander:make_lunar_lander_fuel_bonus_clipped_env",
        )
    except Exception:  # pragma: no cover - id may already exist in tests
        pass


_register_envs()
