import torch
from torch import nn, Tensor

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))

    def silu(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        h1 = torch.einsum("...i,oi->...o", x, self.w1)
        h3 = torch.einsum("...i,oi->...o", x, self.w3)
        return torch.einsum("...i,oi->...o", self.silu(h1) * h3, self.w2)