from typing import Self

import cvxpy as cp
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from torch import Tensor

from src.sets.interface.convex_set import ConvexSet


class Zonotope(ConvexSet):
    """
    A zonotope is a convex set defined as the sum of a point and a linear
    combination of vectors. It is defined by a center and a set of generators as
    Z = {x | x = center + sum_i lambda_i * generator_i, lambda_i in [-1, 1]}.

    Attributes:
        center: The center of the zonotope.
        generator: The generators of the zonotope.
    """
    center: Tensor
    generator: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self, center: Float[Tensor, "batch_dim dim"],
                 generator: Float[
                     Tensor, "batch_dim dim num_generators"]):
        """
        Initialize the zonotope with the given center and generators.

        Args:
            center: The center of the zonotope.
            generator: The generators of the zonotope.
        """
        super().__init__(*center.shape)
        self.center = center
        self.generator = generator
        self.num_gens = self.generator.shape[2]

    def __iter__(self):
        return iter((self.center, self.generator))

    def __getitem__(self, item) -> Self:
        if isinstance(item, int):
            return Zonotope(self.center[item].unsqueeze(0),
                            self.generator[item].unsqueeze(0))
        elif isinstance(item, Tensor):
            return Zonotope(self.center[item], self.generator[item])
        else:
            raise TypeError(f"Invalid argument type {type(item)}")

    def draw(self, ax=None, **kwargs):
        """
        Draw the zonotope. (Assumes 2D and batch_dim=1)

        Args:
            ax: The matplotlib axis to draw the convex set on.
            kwargs: Additional keyword arguments for drawing.
        """
        if ax is None:
            ax = plt.gca()
        polygon = Polygon(self.vertices().T, fill=False, **kwargs)
        ax.add_patch(polygon)

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the box.

        Returns:
            A tensor of sampled points from the box.
        """
        return self.center + torch.sum(
            self.generator * (
                    2 * torch.rand(self.batch_dim, 1, self.num_gens) - 1),
            dim=2)

    @staticmethod
    def point_containment_constraints(point: cp.Expression | np.ndarray,
                                      center: cp.Expression | np.ndarray,
                                      generator: cp.Expression | np.ndarray) \
            -> list[cp.Constraint]:
        """
        Construct the constraints for a point-zonotope containment problem with
        Z = <c, G> and p for p in Z.

        Based on: Kulmburg, A., Althoff, M., (2021): "On the co-NP-Completeness of the
        Zonotope Containment Problem", Eq. (6).

        Args:
            point: The point to check for containment.
            center: The center of the zonotope.
            generator: The generators of the zonotope.

        Returns:
            The constraints for the point-zonotope containment problem.
        """
        weights = cp.Variable(generator.shape[1])

        pos_constraint = point - center == generator @ weights
        size_constraint = cp.norm(weights, "inf") <= 1
        return [pos_constraint, size_constraint]

    @staticmethod
    def zonotope_containment_constraints(c1: cp.Expression | np.ndarray,
                                         g1: cp.Expression | np.ndarray,
                                         c2: cp.Expression | np.ndarray,
                                         g2: cp.Expression | np.ndarray) \
            -> list[cp.Constraint]:
        """
        Construct the constraints for a zonotope-zonotope containment problem with
        Z_1 = <c1, g1> and Z_2 = <c2, g2> for Z_1 in Z_2.

        Based on: Sadraddini, Sadra, and Russ Tedrake. "Linear encodings for polytope
        containment problems." 2019 IEEE 58th conference on decision and control (CDC).
        IEEE, 2019.

        Args:
            c1: The center of the first zonotope.
            g1: The generators of the first zonotope.
            c2: The center of the second zonotope.
            g2: The generators of the second zonotope.

        Returns:
            The constraints for the zonotope-zonotope containment problem.
        """

        weights = cp.Variable(g2.shape[1])
        mapping = cp.Variable((g2.shape[1], g1.shape[1]))

        shape_constraint = g1 == g2 @ mapping
        pos_constraint = c2 - c1 == g2 @ weights
        size_constraint = cp.norm(cp.hstack([
            mapping,
            cp.reshape(weights, (-1, 1), "C")
        ]), "inf") <= 1

        return [shape_constraint, pos_constraint, size_constraint]

    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"] | ConvexSet) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the zonotope.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if the point is contained in the zonotope, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, Tensor):
            weights = cp.Variable((self.batch_dim, self.num_gens))

            constraints = [
                other.numpy() - self.center.numpy() == self.generator[i].numpy() @
                weights[i]
                for i in range(self.batch_dim)
            ]

            objective = cp.Minimize(cp.norm(weights, "inf", axis=1))
            problem = cp.Problem(objective, constraints)
            problem.solve(solver="CLARABEL")

            if not problem.status == cp.OPTIMAL:
                raise Exception("Could not compute point-zonotope containment problem.")
            return torch.tensor([problem.value <= 1], dtype=torch.bool)

        elif isinstance(other, sets.Ball):
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")
        elif isinstance(other, sets.Box):
            return self.contains(Zonotope(other.center, other.generator))
        elif isinstance(other, sets.Capsule):
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")
        elif isinstance(other, Zonotope):
            """
            CVXPY Problem for now only works non batched
            
            weights = cp.Variable((self.batch_dim, self.num_gens))
            mapping = cp.Variable((self.batch_dim, self.num_gens, other.num_gens))

            shape_constraints = [
                other.generator[i].numpy() == self.generator[i].numpy() @ mapping[i]
                for i in range(self.batch_dim)
            ]

            pos_constraints = [
                self.center[i].numpy() - other.center[i].numpy() == self.generator[
                    i].numpy() @ weights[i]
                for i in range(self.batch_dim)
            ]

            constraints = shape_constraints + pos_constraints

            flat_mapping = cp.reshape(mapping,
                                      (self.batch_dim * self.num_gens, other.num_gens),
                                      "C")
            flat_weights = cp.reshape(weights, (self.batch_dim * self.num_gens, 1), "C")

            stacked = cp.reshape(cp.hstack([flat_mapping, flat_weights]),
                                 (self.batch_dim, self.num_gens, other.num_gens + 1),
                                 "C")

            objective = cp.Minimize(cp.norm(stacked, "inf", axis=(1, 2)))
            """
            weights = cp.Variable(self.num_gens)
            mapping = cp.Variable((self.num_gens, other.num_gens))

            shape_constraint = [
                other.generator[0].numpy() == self.generator[0].numpy() @ mapping
            ]

            pos_constraint = [
                self.center[0].numpy() - other.center[0].numpy() == self.generator[
                    0].numpy() @ weights
            ]

            constraints = shape_constraint + pos_constraint

            objective = cp.Minimize(cp.norm(cp.hstack([
                mapping,
                cp.reshape(weights, (-1, 1), "C")
            ]), "inf"))
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if not problem.status == cp.OPTIMAL:
                raise Exception(
                    f"Could not compute zonotope-zonotope containment problem. {problem.status}")
            return torch.tensor([problem.value <= 1], dtype=torch.bool)
        else:
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the zonotope.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the zonotope, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, sets.Box):
            return other.intersects(self)
        elif isinstance(other, sets.Capsule):
            return other.intersects(self)
        elif isinstance(other, sets.Zonotope):
            # Overapproximation
            return self.box().intersects(other.box())
        elif isinstance(other, sets.Polytope):
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")
        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    def box(self) -> Self:
        """
        Return the smallest axis-aligned box enclosure of the zonotope.
        Packaged as a zonotope.

        Returns:
            The smallest axis-aligned box enclosure of the zonotope.
        """
        from src.sets import Box
        return Box(self.center, self.generator.abs().sum(dim=2).diag_embed())

    def vertices(self) -> Float[Tensor, "{self.batch_dim} {self.dim} num_vert"]:
        """
        Compute vertices for a 2D zonotope.
        """
        # Normalize directions to ensure all generators point "up"
        generators_norm = self.generator[0].clone()
        generators_norm[:, generators_norm[1, :] < 0] *= -1

        # Calculate angles and sort generators by these angles
        angles = torch.arctan2(generators_norm[1, :], generators_norm[0, :])
        angles[angles < 0] += 2 * torch.pi
        indices = torch.argsort(angles)

        # Cumulative sum for vertices following sorted angles
        vert = torch.zeros((2, self.num_gens + 1))
        for i in range(self.num_gens):
            vert[:, i + 1] = vert[:, i] + 2 * generators_norm[:, indices[i]]

        # Shift vertices to center the zonotope around the origin
        xmax = self.generator[0, 0, :].abs().sum()
        ymax = self.generator[0, 1, :].abs().sum()
        vert[0, :] = vert[0, :] - vert[0, :].max() + xmax
        vert[1, :] -= ymax

        # Reflect the upper half to get the lower half (point symmetry)
        lower_half = torch.vstack([
            vert[0, -1] + vert[0, 0] - vert[0, 1:],
            vert[1, -1] + vert[1, 0] - vert[1, 1:]
        ])
        vert = torch.hstack([vert, lower_half])

        return self.center[0].unsqueeze(1) + vert


    def setup_constraints(self):
        """
        Setup the residuals for FSNet solver interface.
        Constructs the A, b, C, d  constraint matrices for the zonotope representation.
        specifically the point containment in the zonotope Z = { x | x = c + Gβ , ||β||∞ <= 1 }
        where c is the center, G is the generator matrix, and β are the generator coefficients.
        
        The equality constraints are Cz = d with z = [y; β] and c is the center of the zonotope
        The inequality constraints are Az <= b with z = [y; β] and b is the box constraints on β.

        """

        batch_dim, dim, num_generators = self.generator.shape

        # for eq constraints Cz = d
        # C with shape (batch_dim, dim, dim + num_generators)
        # d with shape (batch_dim, dim, 1)
        
        self.C = torch.cat([
                torch.eye(dim).expand(batch_dim, dim, dim), 
                -self.generator
            ], dim=2).detach() 
        
        self.d = self.center.detach() 

        # for ineq constraints  Az <= b
        # A with shape (batch_dim, 2 * num_generators, dim + num_generators)
        # b with shape (batch_dim, 2 * num_generators, 1)

        A_half = torch.cat([
                torch.zeros((batch_dim, num_generators, dim)),
                torch.eye(num_generators).expand(batch_dim, num_generators, num_generators)
            ], dim=2)
        
        self.A = torch.cat([A_half, -A_half], dim=1).detach()  
        self.b = torch.ones((batch_dim, 2 * num_generators)).detach()  

    def pre_process_action(self, action):
        """
        
        Pre-process the action to fit the zonotope representation. 
        z = [a; gamma], where a is the action and gamma are the generator coefficients.
        z has shape (batch_size, dim + num_generators)
        Args:
            action: The action to pre-process.
        Returns:
                The pre-processed action.
            """
        batch_dim, _, num_generators = self.generator.shape
        z = torch.nn.functional.pad(action, (0, num_generators)) # to allow for backpropagation through actions outside the zonotope
        return z
    
    def post_process_action(self, action):
        """
        Post-process the action to extract the original action from the zonotope representation.
        z = [a; gamma], where a is the action and gamma are the generator coefficients
        Args:
            action: The action to post-process.
        Returns:
            The post-processed action.
        """
        return action[:, :self.dim]
    