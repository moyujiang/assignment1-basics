import torch
from torch import nn, Tensor
from cs336_basics.rope import RoPE
from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.ffn import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        from cs336_basics.attention import CausalMultiHeadSelfAttention
        from cs336_basics.ffn import SwiGLU
        from cs336_basics.rmsnorm import RMSNorm

        self.self_attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        # Pre-norm self-attention

        attn_output = self.self_attn(self.norm1(x), rope=rope, token_positions=token_positions)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        seq_len = in_indices.shape[-1]
        batch_shape = in_indices.shape[:-1]

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds model context length {self.context_length}.")
        
        x = self.token_embedding(in_indices)
        token_positions = torch.arange(seq_len, device=in_indices.device)
        token_positions = token_positions.view(*([1] * len(batch_shape)), seq_len).expand(*batch_shape, seq_len)

        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits