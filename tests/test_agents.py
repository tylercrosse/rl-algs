import torch

from rl_algs.agents import DQNAgent, DQNConfig, PPOAgent, PPOConfig, ReinforceAgent, ReinforceConfig
from rl_algs.data.replay_buffer import ReplayBufferConfig


def test_dqn_basic_update():
    config = DQNConfig(
        obs_dim=4,
        action_dim=2,
        min_buffer_size=1,
        replay_buffer=ReplayBufferConfig(capacity=10, batch_size=1),
    )
    agent = DQNAgent(config)

    transition = {
        "obs": torch.zeros((1, 4), dtype=torch.float32),
        "action": torch.zeros((1, 1), dtype=torch.int64),
        "reward": torch.tensor([[1.0]], dtype=torch.float32),
        "next_obs": torch.zeros((1, 4), dtype=torch.float32),
        "done": torch.zeros((1, 1), dtype=torch.float32),
    }
    agent.store_transition(transition)
    metrics = agent.sample_and_update()
    assert "loss" in metrics


def test_reinforce_episode():
    agent = ReinforceAgent(ReinforceConfig(obs_dim=4, action_dim=2))
    obs = torch.zeros((1, 4), dtype=torch.float32)
    action = agent.select_action(obs)
    assert action.shape == (1,)
    agent.record_reward(1.0)
    metrics = agent.end_episode()
    assert "loss" in metrics


def test_ppo_rollout_update():
    config = PPOConfig(obs_dim=4, action_dim=2, rollout_length=1, minibatch_size=1, num_epochs=1)
    agent = PPOAgent(config)
    transition = {
        "obs": torch.zeros((1, 4), dtype=torch.float32),
        "action": torch.zeros((1, 1), dtype=torch.int64),
        "reward": torch.tensor([[1.0]], dtype=torch.float32),
        "next_obs": torch.zeros((1, 4), dtype=torch.float32),
        "done": torch.zeros((1, 1), dtype=torch.float32),
    }
    agent.store_transition(transition)
    metrics = agent.sample_and_update()
    assert "loss" in metrics
