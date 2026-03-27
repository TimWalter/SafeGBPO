import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float

SLACK_EPS:float = 1e-2

@jaxtyped(typechecker=beartype)
def closest_point_on_line_segment(start: Float[Tensor, "batch_dim dim"],
                                  end: Float[Tensor, "batch_dim dim"],
                                  point: Float[Tensor, "num_samples batch_dim dim"]) \
        -> Float[Tensor, "num_samples batch_dim dim"]:
    """
    Find the closest point on a line segment to a given point.

    Args:
        start: Start of the line segment.
        end: End of the line segment.
        point: The point to find the closest point to.

    Returns:
        The closest point on the line segment to the given point.
    """
    start = start.unsqueeze(0)
    end = end.unsqueeze(0)

    line = end - start
    distance_to_start = torch.clamp(
        torch.sum((point - start) * line, dim=-1, keepdim=True) / torch.sum(line * line, dim=-1, keepdim=True), 0, 1)
    return start + distance_to_start * line


@jaxtyped(typechecker=beartype)
def shortest_line_segment(s1: Float[Tensor, "batch_dim dim"], e1: Float[Tensor, "batch_dim dim"],
                          s2: Float[Tensor, "batch_dim dim"], e2: Float[Tensor, "batch_dim dim"], ) \
        -> tuple[
            Float[Tensor, "batch_dim dim"],
            Float[Tensor, "batch_dim dim"],
        ]:
    """
    Calculate the shortest line segments to connect two line-segments.
    Calculation is obtained from the analytical solution to the minimisation
    of the squared l2 norm.

    Args:
        s1: The start of the first line segment
        e1: The end of the first line segment
        s2: The start of the second line segment
        e2: The end of the second line segment

    Returns:
        Tuple containing the start and end of the shortest connecting line segment.
    """
    l1 = e1 - s1
    l2 = e2 - s2
    ds = s1 - s2

    alpha = (l1 * l1).sum(dim=-1, keepdim=True)
    beta = (l2 * l2).sum(dim=-1, keepdim=True)
    gamma = (l1 * l2).sum(dim=-1, keepdim=True)
    delta = (l1 * ds).sum(dim=-1, keepdim=True)
    epsilon = (l2 * ds).sum(dim=-1, keepdim=True)

    det = alpha * beta - gamma ** 2

    t1 = torch.clamp((gamma * epsilon - beta * delta) / (det + 1e-10), 0.0, 1.0)
    t2 = torch.clamp((gamma * t1 + epsilon) / (beta + 1e-10), 0.0, 1.0)

    t1 = torch.where((t2 == 0.0) | (t2 == 1.0), torch.clamp((t2 * gamma - delta) / (alpha + 1e-10), 0.0, 1.0), t1)

    c1 = s1 + t1 * l1
    c2 = s2 + t2 * l2

    return c1, c2
