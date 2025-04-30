from typing import Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.distributions.normal import Normal

class Policy(nn.Module):
    """
    Stochastic policy network for the actor in an actor-critic algorithms.

    Attributes:
        device (torch.device): Device where the model is stored
        model (nn.Sequential): Neural network model predicting the mean action
        log_std (torch.nn.Parameter): Log standard deviation of the action distribution
        stochastic (bool): Whether the policy is stochastic or deterministic
        last_dist: Buffering the last distribution to save forward passes.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 stochastic: bool = True,
                 net_arch: list = None,
                 activation_fn: str = "nn.ELU()",
                 layer_norm: bool = True,
                 device: torch.device = None):
        """Initialize the policy network.

        Args:
            input_dim (int): Dimension of the inputs
            output_dim (int): Dimension of the outputs
            stochastic (bool): Whether the policy is stochastic or deterministic
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
        self.stochastic = stochastic
        self.input_dim = input_dim
        self.output_dim = output_dim

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
            nn.Linear(net_arch[-1], output_dim)
        ).to(self.device, dtype=torch.float64)

        self.log_std = 0
        if stochastic:
            self.log_std = torch.nn.Parameter(
                -1.0 * torch.ones(output_dim, dtype=torch.float64, device=self.device))

            self.last_dist: Normal = None

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch_size {self.input_dim}"]) \
            -> Float[Tensor, "batch_size {self.output_dim}"]:
        """
        Sample an action given the state.

        Args:
            x: Observation

        Returns:
            Action
        """
        mean = self.model(x)

        if self.stochastic:
            self.last_dist = Normal(mean, torch.exp(self.log_std))
            action = self.last_dist.rsample()
        else:
            action = mean

        return torch.tanh(action)

    @jaxtyped(typechecker=beartype)
    def predict(self,
                x: Float[Tensor, "batch_size {self.input_dim}"],
                deterministic: bool = True) \
            -> Float[Tensor, "batch_size {self.output_dim}"]:
        """
        Predict an action given the state.

        Args:
            x: Observation
            deterministic: Whether to sample an action or return the mean

        Returns:
            Action
        """
        if deterministic:
            return torch.tanh(self.model(x))
        else:
            return self.forward(x)

    @jaxtyped(typechecker=beartype)
    def log_prob(self, action: Optional[Float[Tensor, "batch_size {self.output_dim}"]],
                 x: Optional[Float[Tensor, "batch_size {self.input_dim}"]] = None,
                 ) -> Float[Tensor, "batch_size"]:
        """
        Calculate the log probability of an action.
        Args:
            action: action
            x: Observation

        Returns:
            Log probability of the action
        """
        if x is not None:
            self.last_dist = Normal(self.model(x), torch.exp(self.log_std))

        eps = torch.finfo(action.dtype).eps

        clamped_action = action.clamp(min=-1.0 + eps, max=1.0 - eps)
        gaussian_actions = 0.5 * (clamped_action.log1p() - (-clamped_action).log1p())
        correction_term = torch.log(1 - clamped_action**2)
        non_boundary = self.last_dist.log_prob(gaussian_actions) - correction_term

        at_boundary = torch.where(action >  0,
                                  torch.log(1 - self.last_dist.cdf(gaussian_actions) + eps),
                                  torch.log(self.last_dist.cdf(-gaussian_actions) + eps)
        )
        mask = torch.isclose(action.abs(), torch.ones_like(action))
        log_prob = torch.where(mask, at_boundary, non_boundary)

        return log_prob.sum(dim=1)

    @jaxtyped(typechecker=beartype)
    def entropy(self,
                x: Optional[Float[Tensor, "batch_size {self.input_dim}"]] = None) -> \
    Float[Tensor, "batch_size"]:
        """
        Calculate the entropy of the action distribution.
        Args:
            x: Observation

        Returns:
            Entropy of the action distribution
        """
        if x is not None:
            self.last_dist = Normal(self.model(x), torch.exp(self.log_std))
        # A bit wrong but there is no analytical solution to the entropy of squashed gaussian
        return self.last_dist.entropy().sum(dim=1)
