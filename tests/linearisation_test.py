import torch

from src.sets.box import Box
from envs.simulators.pendulum import PendulumEnv
from envs.simulators.household import HouseholdEnv


def linearisation_test(env_class, **kwargs):
    env = env_class(num_envs=2, num_steps=1000)
    env.reset()

    constant_mat, state_mat, action_mat, noise_mat = env.linear_dynamics()

    action = torch.ones(env.num_envs, env.action_dim) * 0.1
    lin_next_state = (constant_mat +
                      torch.einsum('bij,bj->bi', state_mat, env.state - env.state) +
                      torch.einsum('bij,bj->bi', action_mat, action) +
                      torch.einsum('bij,bj->bi', noise_mat, env.noise_set.center - env.noise_set.center))
    lin_next_state_generator = torch.matmul(noise_mat, env.noise_set.generator)
    lin_next_state = torch.clamp(lin_next_state, env.state_set.min, env.state_set.max)

    next_state_box = Box(lin_next_state,
                         lin_next_state_generator)

    env.step(action)
    next_state = env.state

    assert next_state_box.contains(next_state).all() or torch.allclose(
        lin_next_state, next_state, rtol=0.01)


def test_pendulum_linearisation():
    linearisation_test(PendulumEnv)

def test_household_linearisation():
    linearisation_test(HouseholdEnv)