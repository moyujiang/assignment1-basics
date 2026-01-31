import torch
from torch import nn, Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    @property
    def weight(self):
        return self.scale

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_sqr = torch.einsum("...i,...i->...", x, x) / x.shape[-1]
        rms = torch.sqrt(mean_sqr + self.eps)
        norm_x = x / rms.unsqueeze(-1)
        result = norm_x * self.scale

        return result.to(in_dtype)
