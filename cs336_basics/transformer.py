import torch
from torch import nn, Tensor
from cs336_basics.rope import RoPE
from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.ffn import SwiGLU, FFNSiLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention and SwiGLU FFN.

    Supports both pre-norm and post-norm variants via `norm_position`.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type!r}")
        self.norm_position = norm_position
        self.self_attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = FFNSiLU(d_model, d_ff, device=device, dtype=dtype)
        if use_rmsnorm:
            self.norm1: nn.Module = RMSNorm(d_model, device=device, dtype=dtype)
            self.norm2: nn.Module = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        if self.norm_position == "pre":
            # Pre-norm: Norm -> Sublayer -> Residual
            attn_output = self.self_attn(self.norm1(x), rope=rope, token_positions=token_positions)
            x = x + attn_output

            ffn_output = self.ffn(self.norm2(x))
            x = x + ffn_output
            return x

        # Post-norm (as in assignment writeup):
        #   z = Norm(x + MHA(x))
        #   y = Norm(z + FFN(z))
        attn_output = self.self_attn(x, rope=rope, token_positions=token_positions)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class TransformerLM(nn.Module):
    """Transformer language model with RoPE positional encodings."""
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope: RoPE | None
        if use_rope:
            self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        else:
            self.rope = None
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                use_rmsnorm=use_rmsnorm,
                norm_position=norm_position,
                ffn_type=ffn_type,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final: nn.Module
        if use_rmsnorm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        seq_len = in_indices.shape[-1]
        batch_shape = in_indices.shape[:-1]

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds model context length {self.context_length}.")
        
        x = self.token_embedding(in_indices)
        token_positions: Tensor | None = None
        rope = self.rope
        if rope is not None:
            token_positions = torch.arange(seq_len, device=in_indices.device)
            token_positions = token_positions.view(*([1] * len(batch_shape)), seq_len).expand(*batch_shape, seq_len)

        for layer in self.layers:
            x = layer(x, rope=rope, token_positions=token_positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits