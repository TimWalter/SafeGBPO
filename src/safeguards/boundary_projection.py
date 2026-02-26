import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped
import torch
import scipy.sparse as sp

from safeguards.interfaces.safeguard import Safeguard, SafeEnv


class BoundaryProjectionSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, env: SafeEnv,
                 regularisation_coefficient: float,
                **kwargs):
        Safeguard.__init__(self, env, regularisation_coefficient)
        self.boundary_layer = None

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
     
        if self.boundary_layer is None:
            cp_action = cp.Parameter(self.action_dim)
            parameters = [cp_action]

            cp_safe_action = cp.Variable(self.action_dim)

            objective = cp.Minimize(cp.sum_squares(cp_action - cp_safe_action))

            constraints = self.feasibility_constraints(cp_safe_action)
            if self.action_constrained:
                constraint, params = self.action_safety_constraints(cp_safe_action)
                constraints += constraint
                parameters += params
            if self.state_constrained:
                constraint, params = self.state_safety_constraints(cp_safe_action)
                constraints += constraint
                parameters += params

            problem = cp.Problem(objective, constraints)
            self.boundary_layer = CvxpyLayer(problem, parameters=parameters, variables=[cp_safe_action])

        parameters = [action] + self.constraint_parameters()
        safe_action = self.boundary_layer(*parameters, solver_args=self.solver_args)[0]

        return safe_action

