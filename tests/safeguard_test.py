import torch

from envs.balance_pendulum import BalancePendulumEnv
from safeguards.ray_mask import RayMaskSafeguard
from safeguards.boundary_projection import BoundaryProjectionSafeguard

torch.set_default_dtype(torch.float64)


def test_boundary_projection():
    env = BalancePendulumEnv(num_envs=2, num_steps=100)
    wrapper = BoundaryProjectionSafeguard(env)

    action = torch.tensor([[0.0], [1.0]], requires_grad=True)
    safe_actions = wrapper.actions(action)

    safe_actions.backward(torch.ones_like(safe_actions))

    assert torch.count_nonzero(action.grad) == 2


def test_ray_mask():
    env = BalancePendulumEnv(num_envs=2, num_steps=100)
    for zonotopic_approximation in [True, False]:
        wrapper = RayMaskSafeguard(env, linear_projection=True, zonotopic_approximation=zonotopic_approximation,
                                   passthrough=False)

        action = torch.tensor([[0.0], [1.0]], requires_grad=True)
        safe_actions = wrapper.actions(action)

        safe_actions.backward(torch.ones_like(safe_actions))

        assert torch.count_nonzero(action.grad) == 2
