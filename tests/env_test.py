import torch

from tasks.navigate_seeker import NavigateSeekerTask
from src.sets.box import Box
from src.tasks.balance_cartpole import BalanceCartPoleTask
from src.tasks.balance_pendulum import BalancePendulumTask
from src.tasks.balance_quadrotor import BalanceQuadrotorTask
from src.tasks.swingup_cartpole import SwingUpCartPoleTask
from src.tasks.navigate_quadrotor import NavigateQuadrotorTask
from src.tasks.envs.cartpole import CartPoleEnv
from src.tasks.envs.pendulum import PendulumEnv
from src.tasks.envs.quadrotor import QuadrotorEnv

def env_test(env_class, **kwargs):
    env = env_class(num_envs=2, render_mode="human", **kwargs)
    obs, _ = env.reset(seed=1)
    generic_action = torch.zeros(2, env.action_space.shape[1])
    generic_action[:, 0] = 1.0
    generic_action[:, 1] = 1.0
    for i in range(100):
        obs, reward, termination, truncation, info = env.step(generic_action)
    env.close()


def linearisation_test(env_class, **kwargs):
    env = env_class(**kwargs)
    env.reset()
    lin_state = env.state[0, :]
    lin_action = torch.zeros(env.action_space.shape[1])
    lin_noise = env.noise.center[0, :]
    constant_mat, state_mat, action_mat, noise_mat = env.linear_dynamics(lin_state,
                                                                         lin_action,
                                                                         lin_noise)

    action = lin_action.unsqueeze(0) + 0.2
    lin_next_state = (constant_mat +
                      state_mat @ (env.state[0, :] - lin_state)
                      + action_mat @ (action[0, :] - lin_action) +
                      noise_mat @ (env.noise.center[0, :] - lin_noise))
    lin_next_state_generator = noise_mat @ env.noise.generator[0, :, :]

    next_state_box = Box(lin_next_state.unsqueeze(0),
                         lin_next_state_generator.unsqueeze(0))

    env.step(action)
    next_state = env.state[0, :]

    assert next_state_box.contains(next_state.unsqueeze(0)) or torch.allclose(
        lin_next_state, next_state, rtol=0.01)


def test_cartpole():
    env_test(CartPoleEnv)


def test_cartpole_linearisation():
    linearisation_test(CartPoleEnv)


def test_pendulum():
    env_test(PendulumEnv)


def test_pendulum_linearisation():
    linearisation_test(PendulumEnv, stochastic=False)
    linearisation_test(PendulumEnv, stochastic=True)

def test_quadrotor():
    env_test(QuadrotorEnv)

def test_quadrotor_linearisation():
    linearisation_test(QuadrotorEnv, stochastic=False)
    linearisation_test(QuadrotorEnv, stochastic=True)

def test_balance_cartpole():
    env_test(BalanceCartPoleTask)

def test_balance_pendulum():
    env_test(BalancePendulumTask)

def test_balance_quadrotor():
    env_test(BalanceQuadrotorTask)

def test_swingup_cartpole():
    env_test(SwingUpCartPoleTask)

def test_navigate_quadrotor():
    env_test(NavigateQuadrotorTask, draw_safe_state_set=True)

def test_navigate_seeker():
    env_test(NavigateSeekerTask, draw_safe_state_set=True)

def test_cartpole_gradient_correctness():
    env = BalanceCartPoleTask(num_envs=2)
    env.reset()
    env.state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    action = torch.tensor([[1.0], [1.0]], requires_grad=True)
    obs, reward, termination, truncation, info = env.step(action)
    reward.sum().backward()

    dforce = env.force_mag
    dtemp = dforce / env.m_t
    dthetaacc = -dtemp / (env.l * (4.0 / 3.0 - env.m_p / env.m_t))
    dxacc = dtemp - dthetaacc * env.m_p * env.l / env.m_t

    dxdot = env.dt * dxacc
    dthetadot = env.dt * dthetaacc
    dx = env.dt * dxdot
    dtheta = env.dt * dthetadot

    x = env.state[0, 0]
    theta = env.state[0, 1]
    xdot = env.state[0, 2]
    thetadot = env.state[0, 3]

    dreward = -2.0 * x * dx * env.cart_position_penalty
    dreward -= 2.0 * xdot * dxdot * env.cart_velocity_penalty
    dreward -= 2.0 * theta * dtheta * env.pole_angle_penalty
    dreward -= 2.0 * thetadot * dthetadot * env.pole_velocity_penalty

    assert action.grad[0, 0] == dreward
    assert action.grad[1, 0] == dreward


def test_cartpole_gradient_cutting():
    env = BalanceCartPoleTask(num_envs=2)

    obs, info = env.reset(seed=0)
    action_1 = torch.tensor([[1.0], [1.0]], requires_grad=True)
    action_2 = torch.tensor([[0.0], [0.0]], requires_grad=True)

    obs, reward_1, termination, truncation, info = env.step(action_1)
    env.clear_computation_graph()
    obs, reward_2, termination, truncation, info = env.step(action_2)
    reward_2.sum().backward()

    print(action_1.grad)
    print(action_2.grad)
    assert action_1.grad is None
    assert action_2.grad is not None
