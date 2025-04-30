import torch


class CoupledTensor:
    """
    Coupled tensor that allows indexing the first dimension with a tensor of indices,
    and receiving an enumerated version in the second dimension.

    This is useful in case of e.g. (timestep, num_envs, obs_dim) where you can index
    the timestep only and receive the respective observations for all environments.

    Attributes:
        tensor: The underlying tensor
        helper_index: A helper index tensor that is used to enumerate the second dimension
    """

    def __init__(self, shape: tuple, dtype, device):
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.helper_index = torch.arange(shape[1], device=device)

    def reset(self):
        if self.tensor.dtype == torch.bool:
            self.tensor.fill_(False)
            self.tensor = self.tensor.detach()
        else:
            self.tensor.fill_(0)
            self.tensor = self.tensor.detach()

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            return self.tensor[index, self.helper_index]
        else:
            return self.tensor[index]

    def __setitem__(self, key, value):
        if isinstance(key, torch.Tensor):
            self.tensor[key, self.helper_index] = value
        else:
            self.tensor[key] = value

    def nonzero(self):
        return self.tensor.nonzero()

    def count_nonzero(self):
        return self.tensor.count_nonzero()

    def sum(self, dim=None):
        return self.tensor.sum(dim=dim)

    def __mul__(self, other):
        return self.tensor * other

    def __add__(self, other):
        return self.tensor + other

    def __sub__(self, other):
        return self.tensor - other
