import math

from rl_algs.utils import make_env


def test_fuel_bonus_wrapper_adds_shaping_bonus():
    env = make_env("LunarLanderFuelBonus-v0", seed=123)
    obs, _ = env.reset()
    _, shaped_reward, terminated, truncated, info = env.step(2)  # fire main engine
    env.close()

    assert "true_reward" in info
    assert "fuel_bonus" in info
    assert shaped_reward == info["true_reward"] + info["fuel_bonus"]
    assert info["fuel_bonus"] > 0.0
    assert math.isfinite(info["true_reward"])
    assert terminated in {False, True}
    assert truncated in {False, True}


def test_clipped_wrapper_limits_bonus_magnitude():
    env_bonus = make_env("LunarLanderFuelBonus-v0", seed=321)
    env_clipped = make_env("LunarLanderFuelBonusClipped-v0", seed=321)

    env_bonus.reset()
    env_clipped.reset()

    _, _, _, _, info_bonus = env_bonus.step(2)
    _, _, _, _, info_clipped = env_clipped.step(2)

    env_bonus.close()
    env_clipped.close()

    assert info_bonus["fuel_bonus"] >= info_clipped["fuel_bonus"]
    assert info_clipped["fuel_bonus"] <= 0.25 + 1e-6
