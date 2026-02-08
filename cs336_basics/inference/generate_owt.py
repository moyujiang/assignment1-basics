#!/usr/bin/env python
"""Generate text samples from an OpenWebText-trained checkpoint.

Usage:
  PYTHONUNBUFFERED=1 uv run python -u -m cs336_basics.inference.generate_owt

Defaults are set up for this repo:
  - checkpoint: checkpoints/owt_checkpoint_final.pt
  - config:     configs/train_openwebtext.json
  - tokenizer:  cs336_basics/tokenizer_output_owt

This script mirrors the TinyStories checkpoint generation workflow, but for OWT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.tokenization.bpe_tokenizer import BPETokenizer
from cs336_basics.training.checkpoint import load_checkpoint
from cs336_basics.inference.generate import generate
from cs336_basics.training.optimization import AdamW
from cs336_basics.nn.transformer import TransformerLM


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate text from owt_checkpoint_final.pt")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/owt_checkpoint_final.pt",
        help="Path to checkpoint (.pt)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="configs/train_openwebtext.json",
        help="Training config JSON to reconstruct the model",
    )
    ap.add_argument(
        "--tokenizer-dir",
        type=str,
        default="cs336_basics/tokenizer_output_owt",
        help="Directory containing vocab.json and merges.txt",
    )

    ap.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Prompt to start generation from",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="How many samples to generate",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="Stop generation when <|endoftext|> is produced.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu/cuda). Default: auto.",
    )

    return ap


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_tokenizer(tokenizer_dir: str) -> tuple[BPETokenizer, int]:
    out_dir = Path(tokenizer_dir)
    tok = BPETokenizer.from_files(
        vocab_filepath=str(out_dir / "vocab.json"),
        merge_filepath=str(out_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tok.token_to_id["<|endoftext|>".encode("utf-8")]
    return tok, eos_id


def build_model_from_config(cfg: dict, device: str) -> TransformerLM:
    return TransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        context_length=int(cfg["context_length"]),
        num_layers=int(cfg["num_layers"]),
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        d_ff=int(cfg["d_ff"]),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        use_rope=not bool(cfg.get("no_rope", False)),
        use_rmsnorm=not bool(cfg.get("no_rmsnorm", False)),
        norm_position=str(cfg.get("norm_position", "pre")),
        ffn_type=str(cfg.get("ffn_type", "swiglu")),
    ).to(device)


def main() -> None:
    args = build_argparser().parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    tokenizer, eos_id = load_tokenizer(args.tokenizer_dir)

    model = build_model_from_config(cfg, device)
    optimizer = AdamW(model.parameters())
    iteration = load_checkpoint(args.checkpoint, model, optimizer)
    model.eval()

    eos_token_id = eos_id if args.stop_on_eos else None

    print("=" * 80)
    print("OWT Generation")
    print("=" * 80)
    print(f"device:      {device}")
    print(f"checkpoint:  {args.checkpoint}")
    print(f"iteration:   {iteration}")
    print(f"config:      {args.config}")
    print(f"tokenizer:   {args.tokenizer_dir}")
    print(
        f"temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens} "
        f"stop_on_eos={args.stop_on_eos} seed={args.seed} num_samples={args.num_samples}"
    )
    print("=" * 80)

    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        torch.manual_seed(args.seed + i)

        out = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            device=device,
        )
        out_ids = out[0].tolist()
        text = tokenizer.decode(out_ids)

        print(f"\n---SAMPLE {i+1}/{args.num_samples}---")
        print(f"prompt_tokens: {len(prompt_ids)} | total_tokens: {len(out_ids)} | new_tokens: {len(out_ids)-len(prompt_ids)}")
        print(text)


if __name__ == "__main__":
    main()
