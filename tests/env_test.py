import torch
import imageio
from torchvision import utils as vutils

from envs.simulators.pendulum import PendulumEnv
from envs.simulators.quadrotor import QuadrotorEnv
from envs.simulators.household import HouseholdEnv
from envs.simulators.cartpole import CartPoleEnv
from envs.simulators.seeker import SeekerEnv

from envs.balance_pendulum import BalancePendulumEnv
from envs.balance_quadrotor import BalanceQuadrotorEnv
from envs.manage_household import ManageHouseholdEnv
from envs.navigate_quadrotor import NavigateQuadrotorEnv
from envs.navigate_seeker import NavigateSeekerEnv
from envs.swingup_cartpole import SwingUpCartPoleEnv

torch.set_default_dtype(torch.float64)


def render_grid(env):
    grid = vutils.make_grid(torch.stack(env.render()) * 255, nrow=int(torch.sqrt(torch.tensor([env.num_envs]))))
    return grid.permute(1, 2, 0).to(torch.uint8).numpy()


def simulator_test(env_class, **kwargs):
    env = env_class(num_envs=2, num_steps=900, **kwargs)
    obs, _ = env.reset()
    frames = [render_grid(env)]
    generic_action = torch.zeros(env.num_envs, env.action_dim)
    for i in range(1000):
        generic_action = generic_action + torch.randn(env.num_envs, env.action_dim) * 0.1
        generic_action = torch.clamp(generic_action, -1, 1)
        obs, reward, termination, truncation, info = env.step(generic_action)
        frames.append(render_grid(env))
    env.close()
    imageio.mimsave(f"{env_class.__name__}.mp4", frames, fps=5)


def test_pendulum():
    simulator_test(PendulumEnv)


def test_quadrotor():
    simulator_test(QuadrotorEnv)


def test_household():
    simulator_test(HouseholdEnv)


def test_cartpole():
    simulator_test(CartPoleEnv)


def test_seeker():
    simulator_test(SeekerEnv)


def test_balance_pendulum():
    simulator_test(BalancePendulumEnv)


def test_balance_quadrotor():
    simulator_test(BalanceQuadrotorEnv)


def test_manage_household():
    simulator_test(ManageHouseholdEnv)


def test_navigate_quadrotor():
    simulator_test(NavigateQuadrotorEnv,
                   num_obstacles=1,
                   min_radius=1.0,
                   max_radius=2.0,
                   draw_safe_state_set=True,
                   )


def test_navigate_seeker():
    simulator_test(NavigateSeekerEnv,
                   num_obstacles=1,
                   min_radius=1.0,
                   max_radius=2.0,
                   draw_safe_action_set=True,
                   polytopic_approach=False,
                   )


def test_swingup_cartpole():
    simulator_test(SwingUpCartPoleEnv)
