from __future__ import annotations

from dataclasses import dataclass


def _matmul_flops(m: int, n: int, p: int) -> int:
    """FLOPs for (m x n) @ (n x p)."""
    return 2 * m * n * p


@dataclass(frozen=True)
class TransformerBlockFlops:
    qkv_proj: int
    attn_scores: int
    attn_weighted_sum: int
    out_proj: int
    ffn: int

    @property
    def total(self) -> int:
        return self.qkv_proj + self.attn_scores + self.attn_weighted_sum + self.out_proj + self.ffn


@dataclass(frozen=True)
class TransformerLMFlops:
    blocks: int
    lm_head: int

    @property
    def total(self) -> int:
        return self.blocks + self.lm_head


def transformer_block_flops(batch_size: int, seq_len: int, d_model: int, num_heads: int, d_ff: int) -> TransformerBlockFlops:
    """FLOPs for one Transformer block (matmul-only)."""
    d_k = d_model // num_heads

    qkv_proj = 3 * _matmul_flops(batch_size * seq_len, d_model, d_model)
    out_proj = _matmul_flops(batch_size * seq_len, d_model, d_model)

    attn_scores = _matmul_flops(batch_size * num_heads * seq_len, d_k, seq_len)
    attn_weighted_sum = _matmul_flops(batch_size * num_heads * seq_len, seq_len, d_k)

    ffn = (
        _matmul_flops(batch_size * seq_len, d_model, d_ff)
        + _matmul_flops(batch_size * seq_len, d_model, d_ff)
        + _matmul_flops(batch_size * seq_len, d_ff, d_model)
    )

    return TransformerBlockFlops(
        qkv_proj=qkv_proj,
        attn_scores=attn_scores,
        attn_weighted_sum=attn_weighted_sum,
        out_proj=out_proj,
        ffn=ffn,
    )


def transformer_lm_flops(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> TransformerLMFlops:
    """FLOPs for full Transformer LM forward pass (matmul-only)."""
    block = transformer_block_flops(batch_size, seq_len, d_model, num_heads, d_ff)
    blocks = num_layers * block.total
    lm_head = _matmul_flops(batch_size * seq_len, d_model, vocab_size)
    return TransformerLMFlops(blocks=blocks, lm_head=lm_head)
