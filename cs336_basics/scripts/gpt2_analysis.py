"""
GPT-2 Parameter and FLOPs Analysis

Based on the actual implementation in cs336_basics:
- Token embedding: Embedding(vocab_size, d_model)
- No position embedding (using RoPE instead)
- Per layer:
  - MHSA: q_proj, k_proj, v_proj, o_proj (all d_model x d_model)
  - FFN: SwiGLU with w1, w2, w3 (w1: d_ff x d_model, w2: d_model x d_ff, w3: d_ff x d_model)
  - RMSNorm x 2: scale parameters (d_model each)
- Final RMSNorm: scale (d_model)
- LM head: Linear(d_model, vocab_size)
"""


def count_parameters(vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int) -> dict:
    """Count trainable parameters in our GPT-2 implementation."""
    
    # Token embedding
    token_emb = vocab_size * d_model
    
    # Per layer parameters
    # MHSA: 4 projection matrices, each d_model x d_model
    mhsa_params = 4 * d_model * d_model
    
    # SwiGLU FFN: w1 (d_ff x d_model), w2 (d_model x d_ff), w3 (d_ff x d_model)
    ffn_params = d_ff * d_model + d_model * d_ff + d_ff * d_model
    ffn_params = 2 * d_ff * d_model + d_model * d_ff  # = 3 * d_ff * d_model
    
    # RMSNorm: 2 per layer (ln1, ln2), each with d_model scale parameters
    norm_params = 2 * d_model
    
    # Per layer total
    per_layer = mhsa_params + ffn_params + norm_params
    
    # Final RMSNorm
    final_norm = d_model
    
    # LM head
    lm_head = d_model * vocab_size
    
    # Total
    total = token_emb + num_layers * per_layer + final_norm + lm_head
    
    return {
        "token_embedding": token_emb,
        "mhsa_per_layer": mhsa_params,
        "ffn_per_layer": ffn_params,
        "norm_per_layer": norm_params,
        "per_layer_total": per_layer,
        "all_layers": num_layers * per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


def matmul_flops(m: int, n: int, p: int) -> int:
    """FLOPs for matrix multiply (m x n) @ (n x p) = 2mnp"""
    return 2 * m * n * p


def count_flops(batch_size: int, seq_len: int, vocab_size: int, num_layers: int, 
                d_model: int, num_heads: int, d_ff: int) -> dict:
    """
    Count FLOPs for forward pass (matrix multiplies only).
    
    Per layer:
    1. QKV projections: 3 x (batch*seq, d_model) @ (d_model, d_model)
    2. Attention scores: (batch*heads*seq, d_k) @ (d_k, seq)
    3. Attention weighted sum: (batch*heads*seq, seq) @ (seq, d_k)
    4. Output projection: (batch*seq, d_model) @ (d_model, d_model)
    5. FFN w1: (batch*seq, d_model) @ (d_model, d_ff)
    6. FFN w3: (batch*seq, d_model) @ (d_model, d_ff)
    7. FFN w2: (batch*seq, d_ff) @ (d_ff, d_model)
    
    LM head: (batch*seq, d_model) @ (d_model, vocab_size)
    """
    d_k = d_model // num_heads
    
    # Per layer
    qkv_proj = 3 * matmul_flops(batch_size * seq_len, d_model, d_model)
    attn_scores = matmul_flops(batch_size * num_heads * seq_len, d_k, seq_len)
    attn_weighted = matmul_flops(batch_size * num_heads * seq_len, seq_len, d_k)
    out_proj = matmul_flops(batch_size * seq_len, d_model, d_model)
    
    # SwiGLU: w1 projection + w3 projection + w2 projection
    ffn_w1 = matmul_flops(batch_size * seq_len, d_model, d_ff)
    ffn_w3 = matmul_flops(batch_size * seq_len, d_model, d_ff)
    ffn_w2 = matmul_flops(batch_size * seq_len, d_ff, d_model)
    ffn_total = ffn_w1 + ffn_w3 + ffn_w2
    
    per_layer = qkv_proj + attn_scores + attn_weighted + out_proj + ffn_total
    
    # LM head
    lm_head = matmul_flops(batch_size * seq_len, d_model, vocab_size)
    
    # Total
    total = num_layers * per_layer + lm_head
    
    return {
        "qkv_proj": qkv_proj * num_layers,
        "attn_scores": attn_scores * num_layers,
        "attn_weighted": attn_weighted * num_layers,
        "out_proj": out_proj * num_layers,
        "ffn": ffn_total * num_layers,
        "per_layer_total": per_layer,
        "all_layers": num_layers * per_layer,
        "lm_head": lm_head,
        "total": total,
    }


def format_number(n: int) -> str:
    """Format large numbers in scientific notation."""
    if n >= 1e12:
        return f"{n/1e12:.3f}T"
    elif n >= 1e9:
        return f"{n/1e9:.3f}B"
    elif n >= 1e6:
        return f"{n/1e6:.3f}M"
    elif n >= 1e3:
        return f"{n/1e3:.3f}K"
    return str(n)


def analyze_model(name: str, vocab_size: int, context_length: int, num_layers: int,
                  d_model: int, num_heads: int, d_ff: int, batch_size: int = 1):
    """Complete analysis of a GPT-2 configuration."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Config: vocab={vocab_size}, ctx={context_length}, L={num_layers}, "
          f"d={d_model}, h={num_heads}, d_ff={d_ff}")
    
    # Parameters
    params = count_parameters(vocab_size, num_layers, d_model, num_heads, d_ff)
    print(f"\n--- Parameters ---")
    print(f"Token embedding:    {format_number(params['token_embedding']):>10}")
    print(f"Per layer:")
    print(f"  MHSA:             {format_number(params['mhsa_per_layer']):>10}")
    print(f"  FFN (SwiGLU):     {format_number(params['ffn_per_layer']):>10}")
    print(f"  RMSNorm (x2):     {format_number(params['norm_per_layer']):>10}")
    print(f"  Layer total:      {format_number(params['per_layer_total']):>10}")
    print(f"All layers:         {format_number(params['all_layers']):>10}")
    print(f"Final RMSNorm:      {format_number(params['final_norm']):>10}")
    print(f"LM head:            {format_number(params['lm_head']):>10}")
    print(f"\nTotal parameters:   {format_number(params['total']):>10} ({params['total']:,})")
    print(f"Memory (fp32):      {params['total'] * 4 / 1e9:.4f} GB")
    
    # FLOPs
    flops = count_flops(batch_size, context_length, vocab_size, num_layers, 
                       d_model, num_heads, d_ff)
    print(f"\n--- FLOPs (batch={batch_size}, seq_len={context_length}) ---")
    print(f"QKV projections:    {format_number(flops['qkv_proj']):>10}")
    print(f"Attention scores:   {format_number(flops['attn_scores']):>10}")
    print(f"Attention weighted: {format_number(flops['attn_weighted']):>10}")
    print(f"Output projection:  {format_number(flops['out_proj']):>10}")
    print(f"FFN (SwiGLU):       {format_number(flops['ffn']):>10}")
    print(f"LM head:            {format_number(flops['lm_head']):>10}")
    print(f"\nTotal FLOPs:        {format_number(flops['total']):>10}")
    
    # Proportions
    print(f"\n--- FLOPs Proportions ---")
    total_flops = flops['total']
    components = {
        'QKV proj': flops['qkv_proj'],
        'Attn scores': flops['attn_scores'],
        'Attn weighted': flops['attn_weighted'],
        'Out proj': flops['out_proj'],
        'FFN': flops['ffn'],
        'LM head': flops['lm_head'],
    }
    for name, value in components.items():
        print(f"{name:>15}: {value/total_flops*100:5.2f}%")
    
    return params, flops


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPT-2 Architecture Analysis")
    print("Based on actual cs336_basics implementation:")
    print("  - RoPE positional encoding (no learned position embeddings)")
    print("  - SwiGLU FFN (3 weight matrices: w1, w2, w3)")
    print("  - RMSNorm (scale parameters only)")
    print("="*80)
    
    # (a) GPT-2 XL
    print("\n" + "="*80)
    print("(a) GPT-2 XL - Parameters and Memory")
    print("="*80)
    params_xl, flops_xl = analyze_model(
        "GPT-2 XL",
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    )
    
    # (b) Matrix multiplies breakdown (already shown above)
    print("\n" + "="*80)
    print("(b) Matrix Multiplies for GPT-2 XL (seq_len=1024)")
    print("="*80)
    print("Per layer (48 layers):")
    print("  1. Q projection:  (1024, 1600) @ (1600, 1600) -> (1024, 1600)")
    print("  2. K projection:  (1024, 1600) @ (1600, 1600) -> (1024, 1600)")
    print("  3. V projection:  (1024, 1600) @ (1600, 1600) -> (1024, 1600)")
    print("  4. Attn scores:   (25*1024, 64) @ (64, 1024) -> (25*1024, 1024)")
    print("  5. Attn weighted: (25*1024, 1024) @ (1024, 64) -> (25*1024, 64)")
    print("  6. Out projection:(1024, 1600) @ (1600, 1600) -> (1024, 1600)")
    print("  7. FFN w1:        (1024, 1600) @ (1600, 6400) -> (1024, 6400)")
    print("  8. FFN w3:        (1024, 1600) @ (1600, 6400) -> (1024, 6400)")
    print("  9. FFN w2:        (1024, 6400) @ (6400, 1600) -> (1024, 1600)")
    print("\nLM head:")
    print("  10. Output:       (1024, 1600) @ (1600, 50257) -> (1024, 50257)")
    
    # (c) Most FLOPs
    print("\n" + "="*80)
    print("(c) Component requiring most FLOPs")
    print("="*80)
    print("From the proportions above, FFN (SwiGLU) requires the most FLOPs (~66.9%),")
    print("followed by QKV projections (~16.7%). Attention computation (scores + weighted)")
    print("accounts for ~7.1% of total FLOPs.")
    
    # (d) Other GPT-2 variants
    print("\n" + "="*80)
    print("(d) GPT-2 Variants Comparison")
    print("="*80)
    
    configs = [
        ("GPT-2 Small", 12, 768, 12, 3072),
        ("GPT-2 Medium", 24, 1024, 16, 4096),
        ("GPT-2 Large", 36, 1280, 20, 5120),
    ]
    
    for name, L, d, h, d_ff in configs:
        analyze_model(name, 50257, 1024, L, d, h, d_ff)
    
    print("\n" + "="*80)
    print("Trend Analysis:")
    print("As model size increases:")
    print("  - FFN proportion increases (small: 49.8% -> XL: 66.9%)")
    print("  - LM head proportion decreases (small: 22.6% -> XL: 3.7%)")
    print("  - Attention scores/weighted proportion decreases slightly")
    print("  - QKV projection proportion increases slightly")
    print("="*80)
    
    # (e) GPT-2 XL with longer context
    print("\n" + "="*80)
    print("(e) GPT-2 XL with context_length=16384")
    print("="*80)
    params_xl_16k, flops_xl_16k = analyze_model(
        "GPT-2 XL (16K context)",
        vocab_size=50257,
        context_length=16384,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    )
    
    print(f"\nComparison with context_length=1024:")
    print(f"  FLOPs ratio: {flops_xl_16k['total'] / flops_xl['total']:.2f}x")
    print(f"  Context scaling: {16384 / 1024:.0f}x")
    print("\nProportional changes:")
    print("  - Attention (scores + weighted) increases significantly (~7% -> ~55%)")
    print("  - FFN decreases (~67% -> ~32%)")
    print("  - QKV projections decrease (~17% -> ~8%)")
    print("  - LM head decreases (~4% -> ~2%)")
    print("\nConclusion: At longer contexts, attention computation (O(n²)) dominates,")
    print("while FFN and projections (O(n)) become relatively less significant.")
