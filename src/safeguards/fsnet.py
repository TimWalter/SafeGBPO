from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Callable
import torch

from safeguards.interfaces.safeguard import Safeguard, SafeActionEnv
from utils import lbfgs


class FSNetSafeguard(Safeguard):
    """
    Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees.
    Projecting an unsafe action to the closest safe action by two inner gradient-based optimisation on the safety
    violations.

    Reference:
    @article{nguyen2025fsnet,
        title={FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees},
        author={Hoang T. Nguyen and Priya L. Donti},
        year={2025},
    }
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: SafeActionEnv,
                 regularisation_coefficient: float,
                 penalty_threshold: float,
                 penalty_coefficient: float,
                 differentiable_steps: int,
                 total_steps: int):
        """
        Args:
            env: Safeguarded environment.
            regularisation_coefficient: Weighting of the regularisation.
            penalty_threshold: Defines the level of safety violation at which additional penalties are incorporated.
            penalty_coefficient: Coefficient for the additional penalty.
            differentiable_steps: How many differentiable feasibility seeking steps to take.
            total_steps: How many total iterations of feasibility seeking.
        """
        Safeguard.__init__(self, env, regularisation_coefficient)

        self.penalty_threshold = penalty_threshold
        self.penalty_coefficient = penalty_coefficient
        self.differentiable_steps = differentiable_steps
        self.non_differentiable_steps = max(0, total_steps - differentiable_steps)

        assert self.action_constrained, "Environment must have a safe action set"

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """
        loss_fn = self.get_safety_violation()
        safe_action_set = self.safe_action_set()
        weights = torch.zeros(action.shape[0], safe_action_set.generator.shape[-1], device=action.device)
        stacked_variable = torch.cat([action, weights], dim=-1)

        safe_action_diff = lbfgs(stacked_variable, loss_fn, self.differentiable_steps)
        with torch.no_grad():
            safe_action_nondiff = lbfgs(safe_action_diff.detach(), loss_fn, self.non_differentiable_steps)
        # Passthrough gradient for the remaining steps
        return (safe_action_diff + (safe_action_nondiff - safe_action_diff).detach())[:, :self.action_dim]


    def get_safety_violation(self) -> Callable[
        [Float[Tensor, "{self.batch_dim} {self.action_dim+self.safe_action_gens}"]],
        Float[Tensor, "{self.batch_dim}"]]:
        """
        Get the safety violation function of the current state.

        Returns:
            Safety violation callable.

        """
        safe_action_set = self.safe_action_set()  # Zonotope

        def safety_violation(stacked_variable):
            action = stacked_variable[..., :self.action_dim]
            weights = stacked_variable[..., self.action_dim:]

            representation = safe_action_set.center + (safe_action_set.generator * weights.unsqueeze(1)).sum(dim=-1)
            representation_error = torch.sum((action - representation)**2, dim=1)
            containment_error = torch.clamp(weights.abs() - 1, min=0).pow(2).sum(dim=1)
            return representation_error + containment_error

        return safety_violation
