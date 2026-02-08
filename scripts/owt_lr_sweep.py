#!/usr/bin/env python
"""Learning rate sweep script for OpenWebText training.

Runs training multiple times with different --max-lr values using
configs/train_openwebtext.json as the base config.

Notes:
- cs336_basics.train will auto-suffix checkpoint_dir and run_name with the LR.
- This script keeps a small results JSON for bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# Global flag to track if interrupt was received
_interrupt_requested = False

DEFAULT_LEARNING_RATES = [3e-4, 4e-4, 6e-4, 8e-4, 1e-3]
DEFAULT_BASE_CONFIG = "configs/train_openwebtext.json"
DEFAULT_RESULTS_FILE = "owt_lr_sweep_results.json"


def format_lr(lr: float) -> str:
    """Format learning rate as string used in directory naming.

    Examples:
      0.0006 -> '6e-4'
      0.0012 -> '1.2e-3'
    """
    if lr >= 1e-3:
        return f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")

    # For values < 1e-3, format as integer coefficient when possible
    import math

    exp = int(math.floor(math.log10(lr)))
    coeff = lr / (10**exp)
    if abs(coeff - int(coeff)) < 1e-6:
        return f"{int(coeff)}e{exp}"
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def parse_lrs(values: list[str] | None) -> list[float]:
    if not values:
        return list(DEFAULT_LEARNING_RATES)
    lrs: list[float] = []
    for v in values:
        # allow: --lrs 3e-4 6e-4 OR --lrs 3e-4,6e-4
        parts = [p for p in v.split(",") if p.strip()]
        for p in parts:
            lrs.append(float(p))
    return lrs


def expected_paths(lr_str: str, base_checkpoint_dir: str, tb_dir: str, run_name: str | None) -> dict:
    """Mirror cs336_basics.train auto naming: checkpoints/<name>_lrX."""
    base = Path(base_checkpoint_dir)
    checkpoint_dir = base.parent / f"{base.name}_lr{lr_str}"

    # cs336_basics.train creates tb dir as <tensorboard_dir>/<run_name>_lrX
    # If run_name is None, it uses run_lrX.
    if run_name:
        tb_run = f"{run_name}_lr{lr_str}"
    else:
        tb_run = f"run_lr{lr_str}"
    tensorboard_dir = Path(tb_dir) / tb_run

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "final_checkpoint": str(checkpoint_dir / "checkpoint_final.pt"),
        "tensorboard_dir": str(tensorboard_dir),
    }


def run_training(
    *,
    lr: float,
    base_config: str,
    min_lr: float | None,
    extra_args: list[str],
) -> dict:
    """Run one training job."""
    lr_str = format_lr(lr)

    cmd = [
        sys.executable,
        "-m",
        "cs336_basics.train",
        "--config",
        base_config,
        "--max-lr",
        str(lr),
    ]

    if min_lr is not None:
        cmd += ["--min-lr", str(min_lr)]

    cmd += extra_args

    print("=" * 80)
    print(f"Starting training with max_lr={lr} (lr_str={lr_str})")
    if min_lr is not None:
        print(f"min_lr={min_lr}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    global _interrupt_requested
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=None, stderr=None)

    try:
        while process.poll() is None:
            if _interrupt_requested:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return_code = -2
                break
            time.sleep(0.1)
        else:
            return_code = process.returncode
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return_code = -2
        _interrupt_requested = True

    elapsed_time = time.time() - start_time

    if return_code == -2 or return_code == 130 or _interrupt_requested:
        status = "interrupted"
        error_msg = "Training interrupted by user (Ctrl+C)"
    elif return_code == 0:
        status = "completed"
        error_msg = None
    else:
        status = "failed"
        error_msg = f"Training failed with return code {return_code}"

    return {
        "lr": lr,
        "lr_str": lr_str,
        "min_lr": min_lr,
        "status": status,
        "elapsed_time": elapsed_time,
        "error": error_msg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWebText learning rate sweep")
    parser.add_argument("--config", default=DEFAULT_BASE_CONFIG, help="Base JSON config path")
    parser.add_argument(
        "--lrs",
        nargs="*",
        default=None,
        help="Learning rates. Example: --lrs 3e-4 6e-4 OR --lrs 3e-4,6e-4",
    )
    parser.add_argument(
        "--min-lr-mode",
        choices=["none", "ratio"],
        default="ratio",
        help="How to set min_lr: 'ratio' uses max_lr/10; 'none' leaves default from train.py/config",
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS_FILE, help="Results JSON output")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to cs336_basics.train after '--'. Example: --extra --device cpu",
    )

    args = parser.parse_args()

    lrs = parse_lrs(args.lrs)
    results_path = Path(args.results)

    # Load base config to derive expected dirs (for convenience in results)
    with open(args.config) as f:
        base_cfg = json.load(f)

    base_checkpoint_dir = str(base_cfg.get("checkpoint_dir", "checkpoints/openwebtext"))
    tb_dir = str(base_cfg.get("tensorboard_dir", "runs/openwebtext"))
    run_name = base_cfg.get("run_name")

    print("=" * 80)
    print("OpenWebText LR Sweep")
    print("Base config:", args.config)
    print("Learning rates:", lrs)
    print("Results file:", str(results_path))
    print("=" * 80)

    global _interrupt_requested
    _interrupt_requested = False

    def signal_handler(_sig, _frame):
        global _interrupt_requested
        _interrupt_requested = True
        print("\n" + "=" * 80)
        print("Interrupt received. Will stop after current run...")
        print("=" * 80)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load existing results if present (append)
    results: list[dict] = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
        except Exception:
            results = []

    # Avoid duplicate runs if already completed
    completed_lr = {r.get("lr") for r in results if r.get("status") == "completed"}

    for i, lr in enumerate(lrs, 1):
        if _interrupt_requested:
            break
        if lr in completed_lr:
            print(f"[{i}/{len(lrs)}] Skipping lr={lr} (already completed in results)")
            continue

        min_lr = None
        if args.min_lr_mode == "ratio":
            min_lr = lr / 10

        # Make sure PYTHONUNBUFFERED to get live logs in SLURM/tmux
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        # Run with env by temporarily setting for subprocess via extra args not possible;
        # easiest: rely on env already set (run_owt_lr.sh sets it). Still ok locally.
        result = run_training(lr=lr, base_config=args.config, min_lr=min_lr, extra_args=args.extra)

        # Add expected artifacts paths
        lr_str = result["lr_str"]
        result.update(expected_paths(lr_str, base_checkpoint_dir, tb_dir, run_name))

        results.append(result)
        results_path.write_text(json.dumps(results, indent=2))

        print(f"[{i}/{len(lrs)}] Done: lr={lr} status={result['status']} time={result['elapsed_time']:.1f}s")

        if _interrupt_requested:
            break

    print("=" * 80)
    print("Sweep complete.")
    print("Saved:", str(results_path))
    print("=" * 80)


if __name__ == "__main__":
    main()
