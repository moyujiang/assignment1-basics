"""Training script for Transformer language model with configurable hyperparameters."""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimization import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import load_dataset, get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train(args):
    """Main training loop with periodic evaluation and checkpointing."""
    
    # Initialize Weights & Biases if enabled
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Install with: pip install wandb")
    
    # Setup device
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load datasets with memory mapping
    print(f"Loading datasets...")
    train_data = load_dataset(args.train_data, args.vocab_size, use_mmap=True)
    val_data = load_dataset(args.val_data, args.vocab_size, use_mmap=True)
    print(f"Train size: {len(train_data):,} tokens | Val size: {len(val_data):,} tokens")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    print(f"\nStarting training...")
    model.train()
    
    for iteration in range(start_iter, args.max_iters):
        t0 = time.time()
        
        # Update learning rate
        lr = get_lr_cosine_schedule(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        dt = time.time() - t0
        
        # Logging
        if iteration % args.log_interval == 0:
            print(f"iter {iteration:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.2f}ms")
            
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/iter_time_ms": dt * 1000,
                }, step=iteration)
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, device)
                    val_logits = model(val_inputs)
                    val_loss = cross_entropy(val_logits, val_targets)
                    val_losses.append(val_loss.item())
            
            val_loss_mean = np.mean(val_losses)
            val_ppl = np.exp(val_loss_mean)
            print(f"[EVAL] iter {iteration:6d} | val_loss {val_loss_mean:.4f} | val_ppl {val_ppl:.2f}")
            
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": val_loss_mean,
                    "val/perplexity": val_ppl,
                }, step=iteration)
            
            model.train()
        
        # Checkpointing
        if args.checkpoint_dir and iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_{iteration}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path}")
    
    # Final checkpoint
    if args.checkpoint_dir:
        final_path = Path(args.checkpoint_dir) / "checkpoint_final.pt"
        save_checkpoint(model, optimizer, args.max_iters, final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")
    
    # Finish wandb run
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Transformer language model")
    
    # Data
    parser.add_argument("--train-data", type=str, help="Path to training data (.npy)")
    parser.add_argument("--val-data", type=str, help="Path to validation data (.npy)")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    
    # Model architecture
    parser.add_argument("--context-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=3072, help="FFN hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    # Optimizer
    parser.add_argument("--max-lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-iters", type=int, default=100000, help="Maximum iterations")
    parser.add_argument("--warmup-iters", type=int, default=2000, help="Warmup iterations")
    
    # Logging & checkpointing
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N iterations")
    parser.add_argument("--eval-iters", type=int, default=20, help="Number of validation batches")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Save checkpoint every N iters")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cs336-transformer", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    
    # Config file support
    parser.add_argument("--config", type=str, help="Load config from JSON file")
    
    # First parse to check if config file is provided
    args, _remaining = parser.parse_known_args()
    
    # Load config from file if specified and set defaults
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        parser.set_defaults(**config)
    
    # Now parse all arguments (config defaults + command line overrides)
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.train_data or not args.val_data or not args.vocab_size:
        parser.error("--train-data, --val-data, and --vocab-size are required (either via --config or command line)")
    
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:25s} : {value}")
    print("=" * 80)
    
    train(args)


if __name__ == "__main__":
    main()
