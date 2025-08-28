from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped


class ValueFunction(nn.Module):
    """
    Value function that estimates the value of a state.
    Critic in actor-critic learning_algorithms.

    Attributes:
        model: Neural network model that estimates the value
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 input_dim: int,
                 net_arch: list = None,
                 activation_fn: str = "nn.ELU()",
                 layer_norm: bool = True):
        """Initialize the value function.

        Args:
            input_dim: Dimension of the inputs
            net_arch: List of integers specifying the hidden layer sizes
            activation_fn: Activation function to use
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]

        net_arch = [input_dim] + net_arch
        self.model = nn.Sequential(
            *[
                layer
                for i in range(0, len(net_arch) - 1)
                for layer in (
                                 nn.Linear(net_arch[i], net_arch[i + 1]),
                                 eval(activation_fn)) +
                             ((nn.LayerNorm(net_arch[i + 1]),) if layer_norm else ())
            ],
            nn.Linear(net_arch[-1], 1)
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, " batch_size obs_dim"],
                a: Optional[Float[Tensor, "batch_size action_dim"]] = None) \
            -> Float[Tensor, "batch_size 1"]:
        """
        Obtain the estimated value of a state.

        Args:
            x: State
            a: Action

        Returns:
            Value of the state
        """
        if a is not None:
            x = torch.cat([x, a], dim=1)
        return self.model(x)
