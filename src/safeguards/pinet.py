import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import time
import os
from torch.func import jacrev, vmap
import torch.nn.functional as F

from safeguards.interfaces.safeguard import Safeguard, SafeEnv
import src.sets as sets

class PinetSafeguard(Safeguard):

    """
    Differentiable safeguard based on ADMM projection onto polyhedral safe sets.

    This class projects actions onto the safe set:

        argmin_x ||x - a||^2  subject to  A x <= b

    using a lifted ADMM / Douglas–Rachford splitting scheme, and supports
    backpropagation via implicit differentiation of the converged fixed point.

    The backward pass does NOT backpropagate through the unrolled ADMM iterations,
    but instead differentiates the fixed-point equation.

    See `_ProjectImplicitFn` for details of the implicit backward implementation.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        env: SafeEnv,
        regularisation_coefficient: float,
        n_iter_admm: int,
        n_iter_bwd: int,
        sigma: float = 1.0,
        omega: float = 1.3,
        fpi: bool = False,
        store_trajectory = False,
        **kwargs
    ):
        super().__init__(env, regularisation_coefficient)

        self.n_iter_admm = n_iter_admm
        self.n_iter_bwd = n_iter_bwd

        self.sigma = sigma
        self.omega = omega
        self.fpi = fpi        
        self.save_dim = False
        
        # Visualization #
        self.store_trajectory = store_trajectory
        self.trajectory = []

        if not self.env.safe_action_polytope:
            raise Exception("Polytope attribute has to be True")
        if not hasattr(self.env, "safe_action_set"):
            raise Exception("Environment must implement safe_action_set method")
    
    @jaxtyped(typechecker=beartype)
    def safeguard(
        self,
        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
    ) -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        
        """
        Project actions onto the safe set using ADMM.

        Args:
            action: Raw action tensor.

        Returns:
            Safeguarded action.
        """

        action = action.unsqueeze(2)

        # Linear constraints A x <= b
        constraints = self.env.safe_action_set()
        A, b = constraints.A, constraints.b

        if not self.save_dim:
            # Save frequently used variables
            Bbatch, m, D = A.shape            
            self.negI = -torch.eye(m, device=A.device).unsqueeze(0).repeat(Bbatch, 1, 1)
            self.beq = torch.zeros(Bbatch, m, 1, device=A.device)
            self.save_dim = True

        y_safe = self._run_projection(
            action=action,
            A=A, 
            b=b 
        )

        return y_safe
    
    def _run_admm(self, sk, y_raw, steps):

        """
        Implement Douglas-Rachford algorithm to project into
        the intersection of the hyperplane and box constraints.
        """

        D = self.action_dim
        sk_iter = sk.clone()

        y_raw_D = y_raw[:, :D, :]
        denom = 1 / (1 + 2 * self.sigma)
        addition = 2 * self.sigma * y_raw_D
        if self.store_trajectory:
            self.trajectory.append(self.eq.project(sk_iter))
        
        for _ in range(steps):
            zk = self.eq.project(sk_iter)
            reflect = 2 * zk - sk_iter
            numerator = addition + reflect[:, :D, :]
            reflect[:, :D, :] = numerator * denom
            tk = self.box_project(reflect)
            sk_iter = sk_iter + self.omega * (tk - zk)
            if self.store_trajectory:
                 self.trajectory.append(self.eq.project(sk_iter))

        return sk_iter

    def _elevate(self, x: Tensor, A: Tensor) -> Tensor:
        """
        Lift primal variable into augmented ADMM space.
        """
        Ax = torch.bmm(A, x)
        return torch.cat([x, Ax], dim=1)

    def _run_projection(
        self,
        action: Tensor,
        A: Tensor,
        b: Tensor,
    ) -> Tensor:
        """
        Run ADMM projection with chosen backward method.

        Returns:
            Safe action tensor.
        """
        Bbatch, m, D = A.shape
        total_dim = D + m

        Aeq = torch.cat([A, self.negI], dim=2)

        ub = torch.cat([
            torch.full((Bbatch, D, 1),  torch.inf, device=A.device, dtype=A.dtype),
            b.unsqueeze(2)
        ], dim=1)

        self.eq = sets.Hyperplane(Aeq, self.beq)
        self.box_project = lambda y: torch.clamp(y, None, ub)

        y_safe = _ProjectImplicitFn.apply(
            self._elevate(action, A),
            self._run_admm,
            self.eq.project,
            int(D),
            total_dim,
            self.n_iter_admm,
            self.n_iter_bwd,
            self.fpi,
        )

        return y_safe

class _ProjectImplicitFn(torch.autograd.Function):

    """
    Custom autograd Function implementing implicit differentiation
    through ADMM iterations.
    """

    @staticmethod
    def forward(ctx, yraw,
                step_iteration, step_final, og_dim, dim_lifted, n_iter, n_iter_bwd, fpi):
        
        """
        Forward pass of through n_iter ADMM iterations + Final projection into the hyperplane set.
        """
        
        sK = torch.zeros_like(yraw)
        with torch.no_grad():
            sK = step_iteration(sK, yraw, n_iter)

        y = step_final(sK).detach()

        ctx.save_for_backward(sK.clone().detach(), yraw.clone().detach())
        ctx.step_iteration = step_iteration
        ctx.step_final = step_final
        ctx.n_iter_bwd = n_iter_bwd
        ctx.dim_lifted = dim_lifted
        ctx.fpi = fpi

        return y[:, :og_dim].squeeze(2)

    @staticmethod
    def backward(ctx, grad_y):
        """
        Compute gradient with respect to the ADMM input using implicit differentiation.

        This method backpropagates through a converged ADMM fixed point without
        explicitly forming or inverting the Jacobian of the ADMM iteration. Instead,
        it solves a linear system involving the Jacobian-vector product of the ADMM
        operator using fixed-point iteration or Anderson-style updates.

        Parameters
        ----------
        ctx : torch.autograd.FunctionContext
            Autograd context containing saved tensors, ADMM operators, and solver
            configuration.
        grad_y : torch.Tensor
            Upstream gradient of shape (batch, out_dim).

        Returns
        -------
        grad_yraw : torch.Tensor
            Gradient with respect to the ADMM input `yraw`.
        None
            No gradients are returned for the other forward inputs.
        """

        sK, yraw = ctx.saved_tensors
        step_iteration = ctx.step_iteration
        step_final = ctx.step_final
        n_iter_bwd = ctx.n_iter_bwd
        dim_lifted = ctx.dim_lifted
        fpi = ctx.fpi

        batch, out_dim = grad_y.shape
        grad_y = grad_y.unsqueeze(2).clone().detach()

        # Project grad_y into lifted space
        y_for_vjp = torch.cat([
            grad_y,
            torch.zeros(batch, dim_lifted - out_dim, 1, device=grad_y.device)
        ], dim=1)

        sK_bwd = sK.requires_grad_(True)
        yraw_bwd = yraw.requires_grad_(True)

        # d a_safe / d sK
        with torch.enable_grad():
            y_final = step_final(sK_bwd)

        vjp = torch.autograd.grad(
            outputs=y_final,
            inputs=sK_bwd,
            grad_outputs=y_for_vjp,
            retain_graph=False,
            allow_unused=False
        )[0]

        # Solve system for g in (I - (dADMM/ds_k)^T) g = d L/d a_safe · d a_safe / d s_k
        def iteration_vjp(v):
            with torch.enable_grad():
                admm_plus = step_iteration(sK_bwd, yraw_bwd, 1)
            return torch.autograd.grad(
                outputs=admm_plus,
                inputs=sK_bwd,
                grad_outputs=v,
                retain_graph=False,
                allow_unused=False
            )[0]

        if fpi:
            g = torch.zeros_like(vjp)
            for _ in range(n_iter_bwd):
                g = iteration_vjp(g) + vjp
        else:
            g = vjp.clone()
            for _ in range(n_iter_bwd):
                g = g + 0.2 * (vjp - (g - iteration_vjp(g)))
        
        # d L / d a = g^T · d ADMM / d a
        def final_fn(y_in):
            with torch.enable_grad():
                s = step_iteration(sK_bwd.detach(), y_in, 1)
            return step_final(s)

        _, grad_yraw = torch.autograd.functional.jvp(final_fn, yraw_bwd, g)

        return grad_yraw, None, None, None, None, None, None, None, None, None, None