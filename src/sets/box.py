import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from torch import Tensor

from src.sets.interface.convex_set import ConvexSet
from src.sets.zonotope import Zonotope


class Box(Zonotope):
    """
    A box is similar to a zonotope with the only additional requirements that the
    generators are orthogonal and that there are exactly dim generators.

    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 generator: Float[Tensor, "batch_dim dim dim"]):
        """
        Initialize the box with the given center and half edges.

        Args:
            center: The center of the ball.
            generator: The half edges of the box.
        """
        super().__init__(center, generator)

        self.edge_len = torch.norm(self.generator, dim=1)
        # Ensures that even lower-dimensional boxes are represented by a full
        # orthogonal basis
        if (self.edge_len == 0).any():
            u, s, vt = torch.svd(self.generator)
            self.edge_dir = u @ vt
        elif generator.isinf().any():  # if you have infinite bounds I expect axis alignment
            self.edge_dir = torch.diag_embed(torch.ones(self.batch_dim, self.dim))
        else:
            self.edge_dir = self.generator / self.edge_len.unsqueeze(1)
        self.edge_len[self.edge_len == 0] = 1.0e-9  # for numerical stability
        self.generator[self.generator.isinf()] = torch.finfo().max

    def draw(self, ax=None, **kwargs):
        """
        Draw the box. (Assumes 2D and batch_dim=1)

        Args:
            ax: The matplotlib axis to draw the convex set on.
            kwargs: Additional keyword arguments for drawing.
        """
        if ax is None:
            ax = plt.gca()

        corners = torch.zeros(4, 2) + self.center[0]
        corners[0] += self.generator[0, :, 0] + self.generator[0, :, 1]
        corners[1] += self.generator[0, :, 0] - self.generator[0, :, 1]
        corners[2] += -self.generator[0, :, 0] - self.generator[0, :, 1]
        corners[3] += -self.generator[0, :, 0] + self.generator[0, :, 1]

        polygon = Polygon(corners.numpy(), fill=False, **kwargs)
        ax.add_patch(polygon)

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the box.

        Returns:
            A tensor of sampled points from the box.
        """
        return self.center + torch.sum(
            self.generator * (2 * torch.rand(self.batch_dim, 1, self.dim) - 1), dim=2)

    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"] | ConvexSet) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the box.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if the point is contained in the box, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, Tensor):
            center = other - self.center
            projection_len = torch.sum(center.unsqueeze(2) * self.edge_dir, dim=1).abs()
            return (projection_len <= self.edge_len).all(dim=1)
        elif isinstance(other, sets.Ball):
            center = other.center - self.center
            projection_len = torch.sum(center.unsqueeze(2) * self.edge_dir, dim=1).abs()
            return (projection_len + other.radius <= self.edge_len).all(dim=1)
        elif isinstance(other, sets.Box) or isinstance(other, sets.Zonotope):
            center = other.center - self.center
            projection_len = torch.sum(center.unsqueeze(2) * self.edge_dir, dim=1).abs()
            supports = (self.edge_dir.unsqueeze(2) * other.generator.unsqueeze(3)).sum(dim=1).abs().sum(dim=1)
            return (projection_len + supports <= self.edge_len).all(dim=1)
        elif isinstance(other, sets.Capsule):
            return self.contains(sets.Ball(other.start, other.radius)) & \
                self.contains(sets.Ball(other.end, other.radius))
        else:
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the box.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the box, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, Box):
            center = other.center - self.center
            projection_len = torch.sum(center.unsqueeze(2) * self.edge_dir, dim=1).abs()
            supports = (self.edge_dir.unsqueeze(2) * other.generator.unsqueeze(3)).sum(
                dim=1).abs().sum(dim=1)
            other_projection_len = torch.sum(center.unsqueeze(2) * other.edge_dir,
                                             dim=1).abs()
            other_supports = (
                    other.edge_dir.unsqueeze(2) * self.generator.unsqueeze(3)).sum(
                dim=1).abs().sum(dim=1)
            return (projection_len <= self.edge_len + supports).all(dim=1) & \
                (other_projection_len <= other.edge_len + other_supports).all(dim=1)
        elif isinstance(other, sets.Capsule):
            return other.intersects(self)
        elif isinstance(other, sets.Zonotope):
            # Overapproximation
            center = other.center - self.center
            projection_len = torch.sum(center.unsqueeze(2) * self.edge_dir, dim=1).abs()
            supports = (self.edge_dir.unsqueeze(2) * other.generator.unsqueeze(3)).sum(
                dim=1).abs().sum(dim=1)
            other_edge_dir = other.generator / torch.norm(other.generator, dim=1,
                                                          keepdim=True)
            other_edge_len = torch.norm(other.generator, dim=1)
            other_projection_len = torch.sum(center.unsqueeze(2) * other_edge_dir,
                                             dim=1).abs()
            other_supports = (
                    other_edge_dir.unsqueeze(2) * self.generator.unsqueeze(3)).sum(
                dim=1).abs().sum(dim=1)
            return (projection_len <= self.edge_len + supports).all(dim=1) & \
                (other_projection_len <= other_edge_len + other_supports).all(dim=1)

        elif isinstance(other, sets.Polytope):
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def farthest_box_corner(self, point: Float[Tensor, "batch_dim dim"]) \
            -> Float[Tensor, "batch_dim dim"]:
        """
        Find the farthest corner of a box from a given point.

        Args:
            point: The point to find the farthest corner from.

        Returns:
            The farthest corner of the box from the given point.
        """
        dist1 = torch.norm(
            self.center.unsqueeze(2) + self.generator - point.unsqueeze(2),
            dim=1, keepdim=True)
        dist2 = torch.norm(
            self.center.unsqueeze(2) - self.generator - point.unsqueeze(2),
            dim=1, keepdim=True)
        return self.center + torch.sum(
            torch.where(dist1 > dist2, 1, -1) * self.generator, dim=2)
