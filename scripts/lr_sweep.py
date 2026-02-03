#!/usr/bin/env python
"""Learning rate hyperparameter sweep script for TinyStories training.

This script runs training with different learning rates and collects results.
"""

import json
import subprocess
import sys
from pathlib import Path
import time
import signal
import os
import shutil

# Global flag to track if interrupt was received
_interrupt_requested = False

LEARNING_RATES = [8e-3]

# Base config path
BASE_CONFIG = "configs/train_tinystories.json"

# Results file
RESULTS_FILE = "lr_sweep_results.json"


def format_lr(lr):
    """Format learning rate as string."""
    if lr >= 1e-3:
        # For values >= 1e-3, use 1 decimal place to distinguish values like 1.2e-3
        return f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")
    else:
        exp = int(__import__('numpy').log10(lr))
        coeff = lr / (10 ** exp)
        if abs(coeff - int(coeff)) < 1e-6:
            return f"{int(coeff)}e{exp}"
        else:
            return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_training(lr):
    """Run training with given learning rate."""
    lr_str = format_lr(lr)
    min_lr = 1e-5
    
    print("=" * 80)
    print(f"Starting training with max_lr={lr} (min_lr={min_lr})")
    print("=" * 80)
    
    # Build command
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--config", BASE_CONFIG,
        "--max-lr", str(lr),
        "--min-lr", str(min_lr),
        "--tensorboard",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training using Popen for better signal handling
    global _interrupt_requested
    start_time = time.time()
    
    # Use Popen instead of run to have better control over signal handling
    process = subprocess.Popen(cmd, stdout=None, stderr=None)
    
    try:
        # Wait for process to complete, checking interrupt flag periodically
        while process.poll() is None:
            if _interrupt_requested:
                # Interrupt requested, terminate the subprocess
                process.terminate()
                try:
                    process.wait(timeout=5)  # Give it time to terminate
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate
                    process.wait()
                return_code = -2
                break
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        else:
            # Process completed normally
            return_code = process.returncode
    except KeyboardInterrupt:
        # Fallback: if KeyboardInterrupt still occurs, handle it
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return_code = -2
        _interrupt_requested = True
    
    elapsed_time = time.time() - start_time
    
    # Check return code
    if return_code == -2 or return_code == 130 or _interrupt_requested:
        status = "interrupted"
        error_msg = "Training interrupted by user (Ctrl+C)"
    elif return_code == 0:
        status = "completed"
        error_msg = None
    else:
        status = "failed"
        error_msg = f"Training failed with return code {return_code}"
        print(f"Training failed: {error_msg}")
    
    return {
        "lr": lr,
        "lr_str": lr_str,
        "min_lr": min_lr,
        "status": status,
        "elapsed_time": elapsed_time,
        "error": error_msg,
    }


def extract_final_metrics(lr_str):
    """Extract final validation loss from checkpoint or tensorboard logs."""
    # Try to read from checkpoint first
    checkpoint_dir = Path(f"checkpoints/tinystories_lr{lr_str}")
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    tensorboard_dir = Path(f"runs/tinystories") / f"tinystories-4L-512d_lr{lr_str}"
    
    # Record directory paths for later analysis
    # The actual metrics can be extracted later from tensorboard or checkpoints
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "final_checkpoint": str(final_checkpoint) if final_checkpoint.exists() else None,
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir.exists() else None,
    }


def main():
    """Main function to run learning rate sweep."""
    global _interrupt_requested
    
    print("=" * 80)
    print("Learning Rate Hyperparameter Sweep")
    print("=" * 80)
    print(f"Testing {len(LEARNING_RATES)} learning rates: {LEARNING_RATES}")
    print(f"Base config: {BASE_CONFIG}")
    print("=" * 80)
    print()
    
    results = []
    _interrupt_requested = False  # Reset flag at start
    
    def signal_handler(sig, frame):
        """Handle interrupt signal gracefully."""
        global _interrupt_requested
        _interrupt_requested = True
        print("\n\n" + "=" * 80)
        print("Interrupt received! Will save progress after current training completes...")
        print("=" * 80)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        for i, lr in enumerate(LEARNING_RATES, 1):
            print(f"\n[{i}/{len(LEARNING_RATES)}] Testing lr={lr}")
            
            # Check if interrupt was requested before starting new training
            if _interrupt_requested:
                print("\nInterrupt requested. Stopping sweep...")
                break
            
            result = run_training(lr)
            
            # Extract metrics if training completed
            if result["status"] == "completed":
                metrics = extract_final_metrics(result["lr_str"])
                result.update(metrics)
            
            results.append(result)
            
            # Save intermediate results
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nCompleted: {result['status']}")
            print(f"Time: {result['elapsed_time']:.1f}s")
            
            # Check if interrupt was requested after training
            if _interrupt_requested:
                print("\nInterrupt requested. Stopping sweep...")
                break
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt if it still occurs
        _interrupt_requested = True
        print("\nKeyboardInterrupt caught. Saving progress...")
    
    # Save final results if interrupted
    if _interrupt_requested:
        print("\n\n" + "=" * 80)
        print("Interrupt received! Saving progress...")
        print("=" * 80)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Progress saved to: {RESULTS_FILE}")
        print("Exiting (no shutdown after interrupt)...")
        sys.exit(0)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Learning Rate Sweep Summary")
    print("=" * 80)
    for result in results:
        status_icon = "✓" if result["status"] == "completed" else "✗"
        print(f"{status_icon} lr={result['lr']:.0e}: {result['status']} "
              f"({result['elapsed_time']/3600:.2f}h)")
    print("=" * 80)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
