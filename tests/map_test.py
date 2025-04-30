import torch

from tasks.balance_pendulum import BalancePendulumTask
from tasks.wrapper.orthogonal_ray_map import OrthogonalRayMapWrapper
from tasks.wrapper.boundary_projection import BoundaryProjectionWrapper
from tasks.wrapper.zonotope_ray_map import ZonotopeRayMapWrapper


def test_boundary_projection():
    env = BalancePendulumTask(num_envs=2)
    wrapper = BoundaryProjectionWrapper(env, [0., 0.], [0.], [0., 0.])

    action = torch.tensor([[0.0], [1.0]], requires_grad=True)
    safe_actions = wrapper.safety_layer(action)

    safe_actions.backward(torch.ones_like(safe_actions))

    assert torch.count_nonzero(action.grad) == 2


def test_orthogonal_ray_map():
    env = BalancePendulumTask(num_envs=2)
    wrapper = OrthogonalRayMapWrapper(env, [0., 0.], [0.], [0., 0.], False)

    action = torch.tensor([[0.0], [1.0]], requires_grad=True)
    safe_actions = wrapper.safety_layer(action)

    safe_actions.backward(torch.ones_like(safe_actions))

    assert torch.count_nonzero(action.grad) == 2


def test_zonotope_ray_map():
    env = BalancePendulumTask(num_envs=2)
    wrapper = ZonotopeRayMapWrapper(env, [0., 0.], [0.], [0., 0.], num_generators=2,
                                           reuse_safe_set=False, passthrough=False, linear_projection=False)

    action = torch.tensor([[0.0], [1.0]], requires_grad=True)
    safe_actions = wrapper.safety_layer(action)

    safe_actions.backward(torch.ones_like(safe_actions))

    assert torch.count_nonzero(action.grad) == 2

def test_zonotope_ray_map_reuse():
    env = BalancePendulumTask(num_envs=2)
    wrapper = ZonotopeRayMapWrapper(env, [0., 0.], [0.], [0., 0.], num_generators=2,
                                           reuse_safe_set=True, passthrough=False, linear_projection=False)

    action = torch.tensor([[0.0], [1.0]], requires_grad=True)
    safe_actions = wrapper.safety_layer(action)

    safe_actions.backward(torch.ones_like(safe_actions))

    assert torch.count_nonzero(action.grad) == 2