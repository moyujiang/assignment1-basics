#!/usr/bin/env python
"""Batch size hyperparameter sweep script for TinyStories training.

Runs training with different batch sizes while keeping the total number of
training tokens fixed (default: 327,680,000 tokens).

Usage:
  PYTHONUNBUFFERED=1 uv run python -u scripts/batchsize_sweep.py

Notes:
  - Uses --tag to disambiguate runs so checkpoints/logs don't overwrite.
  - Keeps context_length fixed as defined in the base config.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Sweep values
BATCH_SIZES = [16, 32, 64, 128]

# Base config path
BASE_CONFIG = "configs/train_tinystories.json"

# Keep total training tokens fixed (matches default TinyStories config:
# 128 * 256 * 10000 = 327,680,000)
TOTAL_TRAIN_TOKENS = 327_680_000

# Results file
RESULTS_FILE = "batchsize_sweep_results.json"

_interrupt_requested = False


@dataclass
class SweepResult:
    batch_size: int
    context_length: int
    total_train_tokens: int
    max_iters: int
    warmup_iters: int
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    status: str
    elapsed_time_s: float
    error: str | None = None


def _load_base_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _compute_schedule(
    *,
    batch_size: int,
    context_length: int,
    total_train_tokens: int,
    base_config: dict,
) -> dict:
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")

    tokens_per_iter = batch_size * context_length
    if total_train_tokens % tokens_per_iter != 0:
        raise ValueError(
            f"total_train_tokens ({total_train_tokens}) must be divisible by "
            f"batch_size*context_length ({tokens_per_iter})."
        )

    max_iters = total_train_tokens // tokens_per_iter

    # Keep warmup fraction the same as base config (default TinyStories is 500/10000 = 5%).
    base_max_iters = int(base_config.get("max_iters", 10000))
    base_warmup = int(base_config.get("warmup_iters", 500))
    warmup_frac = base_warmup / base_max_iters if base_max_iters > 0 else 0.0
    warmup_iters = max(1, int(round(max_iters * warmup_frac))) if warmup_frac > 0 else 0

    # Scale iteration-based intervals to keep (approximately) constant token-based frequency.
    base_batch_size = int(base_config.get("batch_size", 128))
    base_tokens_per_iter = base_batch_size * context_length

    def interval_from_tokens(key: str, default_iters: int) -> int:
        base_iters = int(base_config.get(key, default_iters))
        base_tokens = base_iters * base_tokens_per_iter
        scaled = int(round(base_tokens / tokens_per_iter))
        return max(1, scaled)

    log_interval = interval_from_tokens("log_interval", 100)
    eval_interval = interval_from_tokens("eval_interval", 1000)
    checkpoint_interval = interval_from_tokens("checkpoint_interval", 2500)

    return {
        "max_iters": int(max_iters),
        "warmup_iters": int(warmup_iters),
        "log_interval": int(log_interval),
        "eval_interval": int(eval_interval),
        "checkpoint_interval": int(checkpoint_interval),
    }


def _run_training(
    *,
    config_path: str,
    batch_size: int,
    context_length: int,
    total_train_tokens: int,
    schedule: dict,
    enable_tensorboard: bool,
    extra_train_args: list[str],
) -> SweepResult:
    tag = f"bs{batch_size}"

    cmd: list[str] = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "cs336_basics.train",
        "--config",
        config_path,
        "--batch-size",
        str(batch_size),
        "--max-iters",
        str(schedule["max_iters"]),
        "--warmup-iters",
        str(schedule["warmup_iters"]),
        "--log-interval",
        str(schedule["log_interval"]),
        "--eval-interval",
        str(schedule["eval_interval"]),
        "--checkpoint-interval",
        str(schedule["checkpoint_interval"]),
        "--tag",
        tag,
    ]

    if enable_tensorboard:
        cmd.append("--tensorboard")

    if extra_train_args:
        cmd.extend(extra_train_args)

    print("=" * 80)
    print(
        "Starting training with "
        f"batch_size={batch_size} | max_iters={schedule['max_iters']} | "
        f"warmup_iters={schedule['warmup_iters']} | tag={tag}"
    )
    print("Command: " + " ".join(cmd))
    print("=" * 80)

    global _interrupt_requested
    start = time.time()
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

    elapsed = time.time() - start

    if return_code in (-2, 130) or _interrupt_requested:
        status = "interrupted"
        error = "Training interrupted"
    elif return_code == 0:
        status = "completed"
        error = None
    else:
        status = "failed"
        error = f"Training failed with return code {return_code}"

    return SweepResult(
        batch_size=batch_size,
        context_length=context_length,
        total_train_tokens=total_train_tokens,
        max_iters=int(schedule["max_iters"]),
        warmup_iters=int(schedule["warmup_iters"]),
        log_interval=int(schedule["log_interval"]),
        eval_interval=int(schedule["eval_interval"]),
        checkpoint_interval=int(schedule["checkpoint_interval"]),
        status=status,
        elapsed_time_s=float(elapsed),
        error=error,
    )


def main() -> None:
    global _interrupt_requested

    parser = argparse.ArgumentParser(description="TinyStories batch size sweep (fixed total tokens)")
    parser.add_argument("--config", type=str, default=BASE_CONFIG, help="Base JSON config")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=BATCH_SIZES,
        help="Batch sizes to sweep (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--total-train-tokens",
        type=int,
        default=TOTAL_TRAIN_TOKENS,
        help="Total training tokens to keep fixed",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=RESULTS_FILE,
        help="Where to write JSON results",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging (do not pass --tensorboard)",
    )

    args, remaining = parser.parse_known_args()
    extra_train_args = list(remaining)
    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]

    config_path = args.config
    batch_sizes = list(args.batch_sizes)
    total_train_tokens = int(args.total_train_tokens)
    results_file = args.results_file
    enable_tensorboard = not args.no_tensorboard

    base_config = _load_base_config(config_path)
    context_length = int(base_config.get("context_length", 256))

    print("=" * 80)
    print("Batch Size Sweep (TinyStories)")
    print("=" * 80)
    print(f"Base config: {config_path}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total train tokens (fixed): {total_train_tokens:,}")
    print(f"Context length (from base config): {context_length}")
    if extra_train_args:
        print(f"Extra train args: {' '.join(extra_train_args)}")
    print("=" * 80)

    results: list[dict] = []
    _interrupt_requested = False

    results_path = Path(results_file)
    if results_path.parent and str(results_path.parent) != ".":
        results_path.parent.mkdir(parents=True, exist_ok=True)

    def signal_handler(_sig, _frame):
        global _interrupt_requested
        _interrupt_requested = True
        print("\n" + "=" * 80)
        print("Interrupt received! Will stop after current run...")
        print("=" * 80)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for i, bs in enumerate(batch_sizes, 1):
        if _interrupt_requested:
            break

        schedule = _compute_schedule(
            batch_size=bs,
            context_length=context_length,
            total_train_tokens=total_train_tokens,
            base_config=base_config,
        )
        print(
            f"\n[{i}/{len(batch_sizes)}] batch_size={bs} -> max_iters={schedule['max_iters']}, "
            f"warmup_iters={schedule['warmup_iters']}"
        )

        res = _run_training(
            config_path=config_path,
            batch_size=bs,
            context_length=context_length,
            total_train_tokens=total_train_tokens,
            schedule=schedule,
            enable_tensorboard=enable_tensorboard,
            extra_train_args=extra_train_args,
        )
        results.append(asdict(res))

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Completed: {res.status} | elapsed={res.elapsed_time_s:.1f}s")
        if _interrupt_requested:
            break

    print("\n" + "=" * 80)
    print("Batch Size Sweep Summary")
    print("=" * 80)
    for r in results:
        print(
            f"batch_size={r['batch_size']:>3} | status={r['status']:<11} | "
            f"max_iters={r['max_iters']:<6} | warmup={r['warmup_iters']:<5} | "
            f"time={r['elapsed_time_s']/3600:.2f}h"
        )
    print("=" * 80)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
