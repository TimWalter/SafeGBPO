from typing import Self
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

from sets.interface.convex_set import ConvexSet
from scipy.spatial import HalfspaceIntersection

class Polytope(ConvexSet):
    """
    Batched H-Polytope representation defined as:
        P_i = { x | A_i x <= b_i } for i in [0, batch_dim)

    Attributes:
        A: Tensor of shape [batch_dim, num_constraints, dim]
        b: Tensor of shape [batch_dim, num_constraints]
    """
    A: Tensor
    b: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        A: Float[Tensor, "batch_dim num_constraints dim"],
        b: Float[Tensor, "batch_dim num_constraints"],
    ) -> None:
        super().__init__(batch_dim=A.shape[0], dim=A.shape[2])
        if A.ndim != 3 or b.ndim != 2:
            raise ValueError("A must be [B, C, D] and b must be [B, C].")
        if A.shape[1] != b.shape[1]:
            raise ValueError("A and b must have same num_constraints.")
        self.A = A
        self.b = b

        self.C, self.d = None, None

        self.batch_dim = A.shape[0]
        self.num_constraints = A.shape[1]
        self.dim = A.shape[2]
        self.device = A.device
        self._centers: list[Tensor | None] = [None] * self.batch_dim

    def __iter__(self):
        return iter((self.A, self.b))

    def __getitem__(self, idx) -> Self:
        """Return a single HPolytope instance or a batched subset."""
        if isinstance(idx, int):
            return Polytope(self.A[idx].unsqueeze(0), self.b[idx].unsqueeze(0))
        elif isinstance(idx, Tensor) or isinstance(idx, slice):
            return Polytope(self.A[idx], self.b[idx])
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

    def vertices(self):

        A = self.A[0].cpu()
        b = self.b[0].cpu()
        n = self.dim
        c = torch.zeros(n + 1, device="cpu")
        c[-1] = -1

        mask = torch.isfinite(b)
        A_i = A[mask]
        b_i = b[mask]

        unpadded_shape = b_i.shape[0]

        A_ub = torch.cat([
            A_i,
            torch.ones(unpadded_shape, 1, device=A_i.device)
        ], dim=1)

        # Works on torch tensors but on cpu only
        res = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_i, 
            bounds=[(None, None)] * n + [(0, None)],
            method="highs"
        )

        if not res.success or res.x[-1] <= 1e-9:
            return np.empty((0, n))

        interior_point = res.x[:-1]

        halfspaces = torch.cat([
            A,
            -b[:, None]
        ], dim=1)

        # QHull does NOT support array API → numpy
        halfspaces_np = halfspaces.detach().cpu().numpy()

        try:
            hs = HalfspaceIntersection(halfspaces_np, interior_point)
            V = hs.intersections
            V = np.reshape(V, (len(V), n))
        except Exception:
            V = np.empty((0, n))

        return self.order_vertices_ccw(V).T

    
    def order_vertices_ccw(self, points):
        if len(points) == 0:
            return points

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)

        transposed = False
        if points.shape[0] == 2:
            points = points.T
            transposed = True

        center = points.mean(dim=0)

        angles = torch.atan2(
            points[:, 1] - center[1],
            points[:, 0] - center[0]
        )

        order = torch.argsort(angles)
        ordered = points[order]

        ordered = ordered.T if transposed else ordered

        return ordered

    def center(self, idx: int | None = None) -> Float[Tensor, "batch_dim dim"]:
        """Computation of the Chebyshev center of an HPolyhedron HP via linear programming.
        Defined as the center with the ball of largest radius contained in HP.

        Raises:
            EmptySetError: Set is empty.
            UnboundedSetError: Set is unbounded. LP could not converge.

        Returns:
            np.ndarray: Chebyshev center.
        """

        B, num_constraints, dim = self.A.shape

        if self._centers[idx] is not None:
            return self._centers[idx]

        # Select valid constraints
        mask = torch.isfinite(self.b[idx])
        A_i = self.A[idx, mask]
        b_i = self.b[idx, mask]

        if mask.sum() == 0:
            raise ValueError("No finite constraints — Chebyshev center undefined")

        b_i = torch.clamp(b_i, min=-1e8)

        norms = torch.linalg.norm(A_i, dim=1)

        if torch.any((norms == 0) & (b_i < 0)):
            raise ValueError("Zero row in A with negative b")
        
        c = torch.zeros(dim + 1, device=self.device)
        c[0] = -1

        row0 = torch.zeros(dim + 1, device=self.device)
        row0[0] = -1

        A_rest = torch.cat([norms[:, None], A_i], dim=1)
        A_ub = torch.vstack([row0, A_rest])

        b_ub = torch.cat([
            torch.zeros(1, device=self.device),
            b_i
        ])

        # Transform to numpy to use scipy linprog

        c_np    = c.detach().cpu()
        A_ub_np = A_ub.detach().cpu()
        b_ub_np = b_ub.detach().cpu()

        # Solve
        res = linprog(
            c=c_np,
            A_ub=A_ub_np,
            b_ub=b_ub_np,
            bounds=[(0, None)] + [(None, None)] * dim,
            method="highs"
        )

        if res.status in (2, 3):
            self._centers[idx] = torch.zeros(dim, device=self.device)
        else:
            self._centers[idx] = torch.tensor(res.x[1:], device=self.device)

        return self._centers[idx]


    def draw(self, ax=None, batch_idx: int = 0, color="blue", **kwargs):
        """Draw a specific 2D polytope in the batch."""
        if self.dim != 2:
            raise NotImplementedError("Can only draw 2D polytopes.")
        if ax is None:
            ax = plt.gca()
        verts = self.vertices()[batch_idx].cpu().numpy()

        polygon = Polygon(verts, closed=True, fill=False, edgecolor=color, **kwargs)
        ax.add_patch(polygon)
        ax.autoscale()
        return ax

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[torch.Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample a point approximately uniformly from the polytope Ax <= b.
        """
        raise NotImplementedError(
                f"Sample method not implemented for Polytope")

    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the zonotope.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the zonotope, False otherwise.
        """
        import sets as sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, sets.Box):
            return other.intersects(self)
        elif isinstance(other, sets.Capsule):
            return other.intersects(self)
        elif isinstance(other, sets.Zonotope):
            # Overapproximation 
            return other.box().intersects(self)
        elif isinstance(other, sets.Polytope):
             raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")
        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    def contains(self, other):
        return NotImplementedError(
                f"Containment check not implemented for {type(other)}")
    
    def setup_constraints(self):
        """
        Setup the equality and inequality residual functions for FSNet.
        The equality constraints are Cy = d which are trivially satisfied for H-Polytopes. C=d= None
        The inequality constraints are Ay <= b.
        Setup detaches A and b from the computation graph for faster computation during FSNet safeguarding
        """
        self.A = self.A.detach()
        self.b = self.b.detach()
        self.C = torch.zeros((self.batch_dim, 1, self.dim), device=self.A.device) 
        self.d = torch.zeros((self.batch_dim, 1), device=self.A.device)