from dataclasses import asdict

import torch

from algorithms.shac import SHAC
from conf.algorithm.shac import SHACConfig
from tasks.envs.cartpole import CartPoleEnv

def initialize():
    env = CartPoleEnv(device="cpu", num_envs=1, max_episode_steps=2)
    env.reset(seed=0)
    config = asdict(SHACConfig())
    config.pop("name")
    config["len_trajectories"] = 3
    config["vf_fit_num_batches"] = 1
    agent = SHAC(env, **config, device="cpu")
    return agent


def test_collect_trajectories():
    agent = initialize()

    initial_observations, info = agent.env.reset()
    initial_target_values = agent.target_value_function(initial_observations).squeeze(
        dim=1)
    agent.buffer.reset(initial_observations, initial_target_values)

    agent.collect_trajectories()

    expected_t = torch.zeros(agent.buffer.num_envs,
                             dtype=torch.int64) + agent.len_trajectories + 1
    assert torch.equal(expected_t, agent.buffer.t)
    assert agent.buffer.observations.count_nonzero() == (
            agent.len_trajectories * agent.num_envs + 2) * agent.obs_dim
    expected_target = agent.target_value_function(
        agent.buffer.observations.tensor.squeeze(dim=1)).data
    assert torch.allclose(agent.buffer.target_values.tensor.data, expected_target.data)
    assert torch.equal(agent.buffer.episode_ends.nonzero(),
                       torch.tensor([[2, 0], [4, 0]]))
    assert torch.equal(agent.buffer.rewards.nonzero(),
                       torch.tensor([[0, 0], [1, 0], [3, 0]]))

    return agent


def test_update_policy():
    agent = test_collect_trajectories()

    policy_loss = agent.update_policy()

    expected_policy_loss = torch.zeros(1)
    expected_policy_loss -= agent.gamma ** 0 * agent.buffer.rewards[0]
    expected_policy_loss -= agent.gamma ** 1 * agent.buffer.rewards[1]
    expected_policy_loss -= agent.gamma ** 2 * agent.buffer.target_values[2]
    expected_policy_loss -= agent.gamma ** 0 * agent.buffer.rewards[3]
    expected_policy_loss -= agent.gamma ** 1 * agent.buffer.target_values[4]
    expected_policy_loss /= 5

    assert torch.allclose(policy_loss, expected_policy_loss)

    return agent


def test_calculate_estimated_values():
    agent = test_update_policy()

    estimated_values, observations = agent.calculate_estimated_values()

    expected_observations = torch.cat([
        agent.buffer.observations[3],
        agent.buffer.observations[1],
        agent.buffer.observations[0],
    ])

    one_step_1 = agent.buffer.rewards[0] + agent.gamma * agent.buffer.target_values[1]
    two_step_1 = agent.buffer.rewards[0] + agent.gamma * agent.buffer.rewards[
        1] + agent.gamma ** 2 * agent.buffer.target_values[2]
    one_step_2 = agent.buffer.rewards[1] + agent.gamma * agent.buffer.target_values[2]
    one_step_3 = agent.buffer.rewards[3] + agent.gamma * agent.buffer.target_values[4]

    expected_estimated_values = torch.cat([
        one_step_3,
        one_step_2,
        (1 - agent.td_weight) * one_step_1 + agent.td_weight * two_step_1,
    ])

    assert torch.equal(observations, expected_observations)
    assert torch.equal(estimated_values, expected_estimated_values)

    return agent, estimated_values, observations


def test_update_value_function():
    agent, estimated_values, observations = test_calculate_estimated_values()
    agent.buffer.t = agent.buffer.t + 4

    previous_value = agent.value_function(observations)
    agent.update_value_function()
    next_value = agent.value_function(observations)

    previous_difference = torch.mean(torch.abs(previous_value - estimated_values))
    next_difference = torch.mean(torch.abs(next_value - estimated_values))

    assert previous_difference > next_difference

    return agent, estimated_values, observations

def test_update_target_value_function():
    agent, estimated_values, observations = test_update_value_function()

    agent.buffer.t = agent.buffer.t + 4

    previous_value = agent.target_value_function(observations)
    agent.update_value_function()
    agent.update_target_value_function()
    next_value = agent.target_value_function(observations)

    previous_difference = torch.mean(torch.abs(previous_value - estimated_values))
    next_difference = torch.mean(torch.abs(next_value - estimated_values))

    assert previous_difference > next_difference