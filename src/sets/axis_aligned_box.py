import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float

from src.sets.box import Box

class AxisAlignedBox(Box):
    """
    AxisAlignedBox is a special case of Box where the generator is diagonal.
    It represents a box aligned with the axes, where the edges are parallel to the coordinate axes.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 generator: Float[Tensor, "batch_dim dim dim"]):
        """
       Initialize the axis aligned box.

       Args:
           center: The center of the box.
           generator: The generator of the box, which must be a diagonal matrix.
       """
        assert torch.equal(generator, torch.diag_embed(
            torch.diagonal(generator, dim1=-2, dim2=-1))), "Generator must be diagonal for AxisAlignedBox."
        generator = torch.positive(generator)
        super().__init__(center, generator)

        self.min = center - torch.diagonal(generator, dim1=-2, dim2=-1)
        self.max = center + torch.diagonal(generator, dim1=-2, dim2=-1)
