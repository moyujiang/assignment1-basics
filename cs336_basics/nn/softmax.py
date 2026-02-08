import torch
from torch import nn, Tensor

def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        x: Float[Tensor, "..."]: Input tensor.
        dim: int: Dimension along which to compute the softmax.

    Returns:
        Float[Tensor, "..."]: Tensor with the same shape as `x` with softmax applied along `dim`.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x