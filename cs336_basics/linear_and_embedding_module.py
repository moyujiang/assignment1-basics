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

    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]