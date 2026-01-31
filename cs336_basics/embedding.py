import torch
from torch import nn, Tensor

class Embedding(nn.Module):
    """Embedding layer that maps token IDs to dense vectors."""
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]