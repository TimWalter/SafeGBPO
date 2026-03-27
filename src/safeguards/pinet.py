import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

from safeguards.interfaces.safeguard import Safeguard, SafeEnv


class PinetSafeguard(Safeguard):
    """
    PInet Optimizing hard-constrained neural networks with orthogonal projection layers.
    Basically boundary projection but computes the projection via ADMM and backprops using the implicit
    function theorem.

    Reference:
    @article{grontasPinetOptimizingHardconstrained2025,
        title=Pinet: {{Optimizing}} Hard-Constrained Neural Networks with Orthogonal Projection Layers},
        author = {Grontas, Panagiotis D. and Terpin, Antonio and Balta, Efe C. and D'Andrea, Raffaello and Lygeros, John},
        year={2025},
    }
    """
    SIGMA: float = 1.0
    OMEGA: float = 1.3
    affine_transformation: Float[Tensor, "batch_dim dim dim"]
    affine_offset: Float[Tensor, "batch_dim dim"]
    s: Float[Tensor, "{self.batch_dim} {self.action_dim+self.safe_action_gens}"]
    action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]

    class ImplicitBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, action, pinet_instance):
            # 1. Forward ADMM pass
            s_0 = torch.zeros(action.shape[0], pinet_instance.action_dim + pinet_instance.safe_action_gens, device=action.device)
            s = pinet_instance.admm(s_0, action, pinet_instance.n_iter)
            z = pinet_instance.project_affine_subspace(s)
            safe_action = z[:, :pinet_instance.action_dim]

            # 2. Save ONLY what is needed for the backward pass in the context
            ctx.pinet = pinet_instance
            ctx.save_for_backward(s, action)

            return safe_action

        @staticmethod
        def backward(ctx, upstream_gradient):
            pinet = ctx.pinet
            s, action = ctx.saved_tensors

            # d_safe_action / d_s
            s = s.requires_grad_(True)
            with torch.enable_grad():
                z = pinet.project_affine_subspace(s)
                safe_action = z[:, :pinet.action_dim]
                grad_z = torch.autograd.grad(safe_action, s, upstream_gradient)[0]

            # Richardson iteration
            action = action.requires_grad_(True)
            grad_s = grad_z.clone()
            for _ in range(pinet.n_iter_bwd):
                with torch.enable_grad():
                    s_next = pinet.admm(s, action, 1)
                    vjp_s = torch.autograd.grad(s_next, s, grad_s)[0]
                grad_s += 0.2 * (grad_z - (grad_s - vjp_s))

            # d_s / d_action
            with torch.enable_grad():
                s_next_act = pinet.admm(s.detach(), action, n_iter=1)
                grad_action = torch.autograd.grad(s_next_act, action, grad_s)[0]

            return grad_action, None

    @jaxtyped(typechecker=beartype)
    def __init__(self, env: SafeEnv, regularisation_coefficient: float, n_iter: int, n_iter_bwd: int):
        super().__init__(env, regularisation_coefficient)

        self.n_iter = n_iter
        self.n_iter_bwd = n_iter_bwd

        # Remove the self.actions override!
        # Remove self.to_cache, self.backward, self.s, etc.

        self.affine_transformation = None
        self.affine_offset = None

        assert self.action_constrained, "Environment must have a safe action set"

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        # Simply apply the custom autograd function here.
        # PyTorch will correctly route gradients through torch.where in the base class.
        return self.ImplicitBackward.apply(action, self)

    def project_affine_subspace(self, x: Float[Tensor, "batch_dim total_dim"]) -> Float[Tensor, "batch_dim total_dim"]:
        batch_size = x.shape[0] # FIX: Dynamic batch size

        # If caching affine_transformation, ensure the cached version matches the current batch size.
        # A simple fix is to check the shape:
        if self.affine_offset is None or self.affine_offset.shape[0] != batch_size:
            safe_action_set = self.env.safe_action_set()

            identity = torch.eye(self.action_dim, device=x.device).expand(batch_size, -1, -1)
            normal = torch.cat([identity, -safe_action_set.generator[:batch_size]], dim=2)
            anchor = safe_action_set.center[:batch_size]

            normal_inv = torch.linalg.pinv(normal)
            identity2 = torch.eye(normal.shape[2], device=x.device).expand(batch_size, -1, -1)
            self.affine_transformation = identity2 - torch.bmm(normal_inv, normal)
            self.affine_offset = torch.sum(normal_inv * anchor.unsqueeze(1), dim=2)

        return torch.sum(self.affine_transformation * x.unsqueeze(1), dim=2) + self.affine_offset

    def project_k1(self, x: Float[Tensor, "{self.batch_dim} {self.safe_action_gens}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Project onto the first factor of the Cartesian Product set K. In this case K1 is the entire Euclidean space,
        such that the projection is the identity.

        Args:
            x: First part of the stacked variable to project.
        Returns:
            Projected variable.
        """
        return x

    def project_k2(self, x: Float[Tensor, "{self.batch_dim} {self.safe_action_gens}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.safe_action_gens}"]:
        """
        Project onto the second factor of the Cartesian Product set K. In this case K2 is the unit box,
        such that the projection is simply clamping.

        Args:
            x: Second part of the stacked variable to project.
        Returns:
            Projected variable.
        """
        return x.clamp(-1, 1)

    def admm(self,
             s: Float[Tensor, "{self.batch_dim} {self.action_dim+self.safe_action_gens}"],
             y_raw: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
             n_iter: int) -> Float[Tensor, "{self.batch_dim} {self.action_dim+self.safe_action_gens}"]:
        """
        Douglas-Rachford algorithm to project into the intersection of the affine subspace and K.

        Args:
            s: Variable state.
            y_raw: Policy output.
            n_iter: Number of iterations

        Returns:
            Projected s.
        """
        s = s.clone()
        for _ in range(n_iter):
            z = self.project_affine_subspace(s)

            t_inp = 2 * z - s
            t = torch.cat([
                self.project_k1((t_inp[:, :self.action_dim] + 2 * self.SIGMA * y_raw) / (1 + 2 * self.SIGMA)),
                self.project_k2(t_inp[:, self.action_dim:])
            ], dim=1)

            s = s + self.OMEGA * (t - z)

        return s

    def to_cache(self):
        cache = [self.affine_transformation.clone(),
                 self.affine_offset.clone(),
                 self.s.clone(),
                 self.action.clone()
                 ]
        self.affine_transformation = None
        self.affine_offset = None
        self.s = None
        self.action = None

        return cache

    def backward(self, ctx, upstream_gradient, *cache):
        """
        Efficiently compute the VJP with the upstream gradient utilising the implicit function theorem

        Args:
            ctx: Context.
            upstream_gradient: Upstream gradient.
            *cache: All the cached tensors in a list.

        Returns:
            Vector-Jacobi product.
        """
        self.affine_transformation, self.affine_offset, s, action = cache

        s = s.requires_grad_(True)
        with torch.enable_grad():
            z = self.project_affine_subspace(s)
            safe_action = z[:, :self.action_dim]
            grad_z = torch.autograd.grad(safe_action, s, upstream_gradient)[0]

        action = action.requires_grad_(True)
        # Richardson iteration
        grad_s = grad_z.clone()
        for _ in range(self.n_iter_bwd):
            with torch.enable_grad():
                s_next = self.admm(s, action, 1)
                vjp_s = torch.autograd.grad(s_next, s, grad_s)[0]
            grad_s += 0.2 * (grad_z - (grad_s - vjp_s))

        with torch.enable_grad():
            s_next_act = self.admm(s.detach(), action, n_iter=1)
            grad_action = torch.autograd.grad(s_next_act, action, grad_s)[0]

        return grad_action, None, None, None
