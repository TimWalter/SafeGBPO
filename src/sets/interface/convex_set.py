from abc import ABC, abstractmethod
from typing import Self


from torch import Tensor, zeros_like, relu
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool

class ConvexSet(ABC):
    """
    Interface for convex set classes. (Batched)
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, batch_dim: int, dim: int):
        """
        Initialize the convex set.

        Args:
            batch_dim: The batch dimension of the convex set.
            dim: The dimension of the convex set.
        """
        self.batch_dim = batch_dim
        self.dim = dim

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __getitem__(self, item) -> Self:
        pass

    @abstractmethod
    def draw(self, ax=None, **kwargs):
        """
        Draw the convex set.

        Args:
            ax: The matplotlib axis to draw the convex set on.
            kwargs: Additional keyword arguments for drawing.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the convex set.

        Returns:
            A tensor of sampled points from the convex set.
        """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"]) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the convex set.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if other is contained in the convex set, False otherwise.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def intersects(self, other) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the convex set.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the convex set, False otherwise.
        """
        pass

    @abstractmethod
    def setup_constraints(self):
        """
        Setup any necessary data structures for computing residuals.
        """
        pass

    def pre_process_action(self, action):
        """
        Pre-process the action for FSNet safeguarding.
        Default implementation does nothing.
        
        Args:
            action: The action to pre-process.  
        Returns:
            The pre-processed action.
        """
        return action
    
    def post_process_action(self, action):
        """
        Post-process the action after FSNet safeguarding.
        Default implementation does nothing.
        Args:
            action: The action to post-process.
        Returns:
            The post-processed action.
        """
        return action
    
    def equality_constraint_violation(self, X: Tensor = None, Y: Tensor = None) -> Tensor:
        """
        Compute the residual for equality constraints Cy = d.
        X exist for compatibility with FSNet interface which can handle input dependent constraints.
        Args:
            X: Optional input tensor (not used for convex sets).
            Y: The tensor to check against the equality constraints.
        Returns:
            The residual tensor for the equality constraints.
        """
        if self.C is None or self.d is None:
            raise AssertionError("Constraint matrices not set up. setup_constraints must be called before equality_constraint_violation")
        return self.C @ Y.unsqueeze(2) - self.d.unsqueeze(2)

    def inequality_constraint_violation(self, X: Tensor= None, Y: Tensor = None) -> Tensor:
        """
        Inequality residuals for Ay <= b constraints.

        Args:
            X: Not used.
            Y: Action tensor of shape (batch_size, action_dim)
        Returns:
            Tensor of shape (batch_size, num_constraints) representing the violation amounts.
        """
        if self.A is None or self.b is None:
            raise AssertionError("Constraint matrices not set up. setup_constraints must be called before inequality_constraint_violation")
        return relu(self.A @ Y.unsqueeze(2) - self.b.unsqueeze(2)) 