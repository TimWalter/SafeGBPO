import torch
from learning_algorithms.components.coupled_tensor import CoupledTensor
from learning_algorithms.components.coupled_buffer import CoupledBuffer


def init_coupled_tensor():
    ct = CoupledTensor(3, 6, 2)
    ct.tensor = torch.arange(3 * 6 * 2).reshape(3, 6, 2).to(torch.get_default_dtype())
    return ct


def test_coupled_tensor_coupled_indexing():
    ct = init_coupled_tensor()
    t = torch.ones(6, dtype=torch.int64)
    t[2] = 2

    retrieval_coupled = ct[t]
    expected_result = torch.tensor([
        [12., 13.],
        [14., 15.],
        [28., 29.],
        [18., 19.],
        [20., 21.],
        [22., 23.]
    ])
    assert retrieval_coupled.shape == torch.Size([6, 2])
    assert torch.allclose(retrieval_coupled, expected_result)
    ct[t] = torch.ones(6, 2)
    assert torch.allclose(ct[t], torch.ones(6, 2))


def test_coupled_tensor_full_indexing():
    ct = init_coupled_tensor()
    retrieval_full = ct[1, 4, 1]
    assert retrieval_full == 21.0
    ct[2, 5, 1] = 100
    assert ct[2, 5, 1] == 100.0


def test_coupled_tensor_coupled_boolean_masking():
    ct = init_coupled_tensor()
    t = torch.ones(6, dtype=torch.int64)
    t[2] = 2

    mask = torch.tensor([True, False, True, False, False, False])
    retrieval_coupled = ct[t[mask], mask]
    expected_result = torch.tensor([
        [12., 13.],
        [28., 29.],
    ])
    assert retrieval_coupled.shape == torch.Size([2, 2])
    assert torch.allclose(retrieval_coupled, expected_result)
    ct[t[mask], mask] = torch.ones(2, 2)
    assert torch.allclose(ct[t[mask], mask], torch.ones(2, 2))


def test_coupled_tensor_nonzero():
    ct = init_coupled_tensor()
    assert torch.allclose(ct.nonzero(), ct.tensor.nonzero())


def test_coupled_tensor_count_nonzero():
    ct = init_coupled_tensor()
    assert ct.count_nonzero() == ct.tensor.count_nonzero()


def test_coupled_tensor_calc():
    ct = init_coupled_tensor()
    assert torch.allclose(ct * 2, ct.tensor * 2)
    assert torch.allclose(ct + 2, ct.tensor + 2)
    assert torch.allclose(ct - 2, ct.tensor - 2)

def test_coupled_tensor_gradients():
    ct = init_coupled_tensor()
    t = torch.ones(6, dtype=torch.int64)

    gradient_tacker = torch.ones((6,2), requires_grad=True)
    ct[t] = gradient_tacker

    loss = torch.zeros(6)
    loss -= (ct[t] * 2).sum(dim=1)

    loss.backward(torch.ones([6]))
    assert gradient_tacker.grad[0,0] == -2.0

def test_coupled_buffer_gradients():

    cb = CoupledBuffer(10, 6, 2, True)

    gradient_tacker_obs = torch.ones((6,2), requires_grad=True)
    gradient_tacker_rew = torch.ones(6, requires_grad=True)
    target_values = torch.ones(6, requires_grad=False)
    episode_ends = torch.zeros(6, dtype=torch.bool, requires_grad=False)

    dummy_obs = torch.zeros_like(gradient_tacker_obs, requires_grad=False)
    dummy_rew = torch.zeros_like(gradient_tacker_rew, requires_grad=False)
    dummy_target = torch.zeros_like(target_values, requires_grad=False)
    dummy_episode_ends = torch.zeros_like(episode_ends, requires_grad=False)
    cb.add(dummy_obs, dummy_rew, dummy_episode_ends, dummy_target)
    cb.add(gradient_tacker_obs, gradient_tacker_rew, episode_ends, target_values)
    cb.add(dummy_obs, dummy_rew, dummy_episode_ends, dummy_target)

    loss = -(cb.observations * 2).sum(dim=[1,2]) - cb.rewards.sum(dim=1)

    loss.backward(torch.ones_like(loss))
    assert gradient_tacker_obs.grad[0,0] == -2.0
    assert gradient_tacker_rew.grad[0] == -1.0

    cb.reset()
    gradient_tacker_obs = torch.ones((6, 2), requires_grad=True)
    gradient_tacker_rew = torch.ones(6, requires_grad=True)
    cb.add(gradient_tacker_obs, gradient_tacker_rew, episode_ends, target_values)

    loss = -(cb.observations * 2).sum(dim=[1, 2]) - cb.rewards.sum(dim=1)

    loss.backward(torch.ones_like(loss))
    assert gradient_tacker_obs.grad[0, 0] == -2.0
    assert gradient_tacker_rew.grad[0] == -1.0

def test_coupled_buffer_episodic_rewards():
    cb = CoupledBuffer(10, 6, 2)

    cb.rewards.tensor = torch.arange(10*6).reshape(10, 6)
    cb.t += 10
    cb.t[3] = 9
    cb.terminals.tensor = torch.zeros(10, 6, dtype=torch.bool)
    cb.terminals.tensor[6, :] = True
    cb.terminals.tensor[8, 2] = True

    result = cb.episodic_rewards()
    print(result)