import torch
from torch import nn, Tensor


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.W = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.trunc_normal_(self.W, std=(2.0 / (in_features + out_features)) ** 0.5)

    @property
    def weight(self):
        return self.W

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...i,oi->...o", x, self.W)

    
