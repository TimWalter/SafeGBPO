from typing import Optional

import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor


class ValueFunction(nn.Module):
    """
    Value function that estimates the value of a state.
    Critic in actor-critic algorithms.

    Attributes:
        device: Device where the model is stored
        model: Neural network model that estimates the value
    """

    def __init__(self,
                 input_dim: int,
                 net_arch: list = None,
                 activation_fn: str = "nn.ELU()",
                 layer_norm: bool = True,
                 device: torch.device = None):
        """Initialize the value function.

        Args:
            input_dim (int): Dimension of the inputs
            net_arch (list): List of integers specifying the hidden layer sizes
            activation_fn (str): Activation function to use
            layer_norm (bool): Whether to use layer normalization
            device (torch.device): Device where the model is stored
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
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
        ).to(self.device, dtype=torch.float64)

    def forward(self, x: Float[Tensor, " batch_size obs_dim"],
                a: Optional[Float[Tensor, "batch_size action_dim"]] = None) \
            -> Float[Tensor, "batch_size 1"]:
        """
        Obtain the estimated value of a state.

        Args:
            x: State, shape [batch_size, obs_dim]
            a: Action, shape [batch_size, action_dim]

        Returns:
            value: Value of the state, shape [batch_size, 1]
        """
        if a is not None:
            x = torch.cat([x, a], dim=1)
        return self.model(x)
