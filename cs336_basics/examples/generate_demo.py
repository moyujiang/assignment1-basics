#!/usr/bin/env python3
"""Example script demonstrating text generation with a language model."""

import torch
from pathlib import Path
from cs336_basics.transformer import TransformerLM
from cs336_basics.generate import generate
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.optimization import AdamW


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> TransformerLM:
    """Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
    
    Returns:
        Loaded TransformerLM model.
    """
    # Load checkpoint to infer model config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from checkpoint (requires knowing config)
    # For now, this is a template - adjust parameters as needed
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=6,
        d_model=384,
        num_heads=6,
        d_ff=1536,
        rope_theta=10000.0,
    ).to(device)
    
    # Create dummy optimizer for loading
    optimizer = AdamW(model.parameters())
    load_checkpoint(checkpoint_path, model, optimizer)
    
    return model


def main():
    """Interactive generation example."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("Language Model Text Generation Demo")
    print("=" * 80)
    
    # Example: Generate with a small model (untrained for demo)
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=3,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        rope_theta=10000.0,
    ).to(device)
    model.eval()
    
    # Create dummy token input (in practice, would use tokenizer)
    # Example: start with token ID 42 (arbitrary choice)
    batch_size = 2
    start_token = torch.full((batch_size, 1), 42, dtype=torch.long, device=device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Example 1: Basic generation
    print("\n" + "=" * 40)
    print("Example 1: Basic Generation (temperature=1.0, no top-p)")
    print("=" * 40)
    output_ids = generate(
        model,
        start_token,
        max_new_tokens=20,
        temperature=1.0,
        top_p=1.0,
        device=device,
    )
    print(f"Generated shape: {output_ids.shape}")
    print(f"Sample output (first sequence): {output_ids[0].tolist()}")
    
    # Example 2: Lower temperature (more deterministic)
    print("\n" + "=" * 40)
    print("Example 2: Lower Temperature (temperature=0.5)")
    print("=" * 40)
    output_ids = generate(
        model,
        start_token,
        max_new_tokens=20,
        temperature=0.5,
        top_p=1.0,
        device=device,
    )
    print(f"Generated shape: {output_ids.shape}")
    print(f"Sample output (first sequence): {output_ids[0].tolist()}")
    
    # Example 3: Higher temperature (more random)
    print("\n" + "=" * 40)
    print("Example 3: Higher Temperature (temperature=2.0)")
    print("=" * 40)
    output_ids = generate(
        model,
        start_token,
        max_new_tokens=20,
        temperature=2.0,
        top_p=1.0,
        device=device,
    )
    print(f"Generated shape: {output_ids.shape}")
    print(f"Sample output (first sequence): {output_ids[0].tolist()}")
    
    # Example 4: Top-p sampling
    print("\n" + "=" * 40)
    print("Example 4: Top-p Sampling (p=0.9)")
    print("=" * 40)
    output_ids = generate(
        model,
        start_token,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9,
        device=device,
    )
    print(f"Generated shape: {output_ids.shape}")
    print(f"Sample output (first sequence): {output_ids[0].tolist()}")
    
    # Example 5: Combined temperature + top-p
    print("\n" + "=" * 40)
    print("Example 5: Temperature 0.7 + Top-p 0.95")
    print("=" * 40)
    output_ids = generate(
        model,
        start_token,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.95,
        device=device,
    )
    print(f"Generated shape: {output_ids.shape}")
    print(f"Sample output (first sequence): {output_ids[0].tolist()}")
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
