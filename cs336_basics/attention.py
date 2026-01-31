import torch
import math
import einops
from torch import nn, Tensor
from cs336_basics.softmax import softmax
from cs336_basics.rope import RoPE

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        Q: Float[Tensor, "... sequence_length d_k"]: Query tensor.
        K: Float[Tensor, "... sequence_length d_k"]: Key tensor.
        V: Float[Tensor, "... sequence_length d_v"]: Value tensor.
        mask: Optional[Float[Tensor, "... sequence_length sequence_length"]]: Attention mask.

    Returns:
        Float[Tensor, "... sequence_length d_v"]: The result of the attention mechanism.
    """
    d_k = Q.shape[-1]
    scores = torch.einsum("...id,...jd->...ij", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return torch.einsum("...ij,...jd->...id", attn_weights, V)

class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""
    
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        nn.init.xavier_uniform_(self.q_proj)
        nn.init.xavier_uniform_(self.k_proj)
        nn.init.xavier_uniform_(self.v_proj)
        nn.init.xavier_uniform_(self.o_proj)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]

        q = torch.einsum("...i,oi->...o", x, self.q_proj)
        k = torch.einsum("...i,oi->...o", x, self.k_proj)
        v = torch.einsum("...i,oi->...o", x, self.v_proj)

        # using einops to reshape for multi-head attention
        q = einops.rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = einops.rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = einops.rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        if rope is not None and token_positions is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        attn = scaled_dot_product_attention(q, k, v, mask=mask)

        attn = einops.rearrange(attn, "... h s d -> ... s (h d)")
        out = torch.einsum("...i,oi->...o", attn, self.o_proj)

        return out
