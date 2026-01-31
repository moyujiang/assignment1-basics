#!/usr/bin/env python
"""Quick training demo with TensorBoard logging on TinyStories data."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.train import train


def main():
    """Run a quick training demo."""
    
    # Check if data exists
    data_dir = Path(__file__).parent.parent.parent / "data" / "tokenized"
    train_path = data_dir / "tinystories_train.uint16.npy"
    val_path = data_dir / "tinystories_valid.uint16.npy"
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Tokenized data not found at {data_dir}")
        print("Please run the tokenization script first:")
        print("  python -m cs336_basics.encode_datasets")
        return
    
    # Get vocab size
    vocab_json = Path(__file__).parent.parent / "tokenizer_output_tinystories" / "vocab.json"
    if vocab_json.exists():
        import json
        with open(vocab_json) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        vocab_size = 10000  # Default
    
    print("=" * 80)
    print("Quick Training Demo - TinyStories 4L-512d")
    print("=" * 80)
    print(f"Dataset: TinyStories")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print("=" * 80)
    print()
    
    # Create args
    class Args:
        pass
    
    args = Args()
    
    # Data
    args.train_data = str(train_path)
    args.val_data = str(val_path)
    args.vocab_size = vocab_size
    
    # Model - 4L-512d as specified
    args.context_length = 256
    args.num_layers = 4
    args.d_model = 512
    args.num_heads = 16
    args.d_ff = 1344
    args.rope_theta = 10000.0
    
    # Optimizer
    args.max_lr = 6e-4
    args.min_lr = 6e-5
    args.weight_decay = 0.1
    args.beta1 = 0.9
    args.beta2 = 0.95
    args.eps = 1e-8
    args.grad_clip = 1.0
    
    # Training - short for demo
    args.batch_size = 32
    args.max_iters = 200  # ~1,638,400 tokens
    args.warmup_iters = 20
    
    # Logging
    args.log_interval = 10
    args.eval_interval = 50
    args.eval_iters = 20
    args.checkpoint_dir = "checkpoints/demo"
    args.checkpoint_interval = 50
    args.resume = None
    
    # System
    import torch
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed = 42
    
    # Logging
    args.tensorboard = True
    args.tensorboard_dir = "runs/demo"
    args.run_name = "quick-demo"
    
    print(f"Device: {args.device}")
    print(f"Model: {args.num_layers}L-{args.d_model}d-{args.num_heads}H (~17M params)")
    print(f"Batch size: {args.batch_size}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Estimated tokens: {args.batch_size * args.max_iters * args.context_length:,}")
    print()
    print("Starting training...")
    print("=" * 80)
    print()
    
    # Run training
    train(args)
    
    print()
    print("=" * 80)
    print("Training complete!")
    print()
    print("To view TensorBoard logs:")
    print(f"  tensorboard --logdir {args.tensorboard_dir}")
    print()
    print("Then open: http://localhost:6006")
    print("=" * 80)


if __name__ == "__main__":
    main()
