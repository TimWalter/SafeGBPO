import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped


class CoupledTensor:
    """
    Coupled tensor that allows indexing the first dimension with a tensor of indices
    and receiving an enumerated version in the second dimension.

    This is useful in case of e.g. (timestep, num_envs, obs_dim) where you can index
    the timestep only and receive the respective observations for all environments.

    Attributes:
        tensor: The underlying tensor
        helper_index: A helper index tensor that is used to enumerate the second dimension
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, *shape, dtype=torch.get_default_dtype()):
        self.tensor = torch.zeros(shape, dtype=dtype)
        self.helper_index = torch.arange(shape[1])

    @jaxtyped(typechecker=beartype)
    def reset(self):
        if self.tensor.dtype == torch.bool:
            self.tensor.fill_(False)
            self.tensor = self.tensor.detach()
        else:
            self.tensor.fill_(0)
            self.tensor = self.tensor.detach()

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index):
        if isinstance(index, Tensor):
            return self.tensor[index, self.helper_index]
        else:
            return self.tensor[index]

    @jaxtyped(typechecker=beartype)
    def __setitem__(self, key, value):
        if isinstance(key, Tensor):
            self.tensor[key, self.helper_index] = value
        else:
            self.tensor[key] = value

    @jaxtyped(typechecker=beartype)
    def nonzero(self):
        return self.tensor.nonzero()

    @jaxtyped(typechecker=beartype)
    def count_nonzero(self):
        return self.tensor.count_nonzero()

    @jaxtyped(typechecker=beartype)
    def sum(self, dim=None):
        return self.tensor.sum(dim=dim)

    @jaxtyped(typechecker=beartype)
    def __mul__(self, other):
        return self.tensor * other

    @jaxtyped(typechecker=beartype)
    def __add__(self, other):
        return self.tensor + other

    @jaxtyped(typechecker=beartype)
    def __sub__(self, other):
        return self.tensor - other
