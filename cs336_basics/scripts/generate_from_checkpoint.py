#!/usr/bin/env python3
"""Generate text from a trained checkpoint.

Example:
  uv run --no-sync python cs336_basics/scripts/generate_from_checkpoint.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --tokenizer tinystories \
    --prompt "Once upon a time," \
    --max-new-tokens 256 \
    --temperature 0.95 \
    --top-p 0.1

Meets assignment deliverable by printing a decoded text dump and token counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.tokenization.bpe_tokenizer import BPETokenizer
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.inference.generate import generate
from cs336_basics.training.checkpoint import load_checkpoint
from cs336_basics.training.optimization import AdamW


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_final.pt")

    ap.add_argument(
        "--tokenizer",
        type=str,
        choices=["tinystories", "owt"],
        default="tinystories",
        help="Which bundled tokenizer_output_* to use.",
    )

    ap.add_argument("--prompt", type=str, default="Once upon a time,")
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
        help="Override device (e.g., cpu/cuda). Default: auto.",
    )

    ap.add_argument(
        "--force-at-least-256",
        action="store_true",
        help=(
            "If --stop-on-eos ends too early (<256 new tokens), rerun once without EOS stopping "
            "to reach >=256 new tokens."
        ),
    )

    return ap


def load_tokenizer(name: str) -> tuple[BPETokenizer, int]:
    if name == "tinystories":
        out_dir = Path("cs336_basics/tokenizer_output_tinystories")
    elif name == "owt":
        out_dir = Path("cs336_basics/tokenizer_output_owt")
    else:
        raise ValueError(f"Unknown tokenizer {name}")

    tok = BPETokenizer.from_files(
        vocab_filepath=str(out_dir / "vocab.json"),
        merge_filepath=str(out_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tok.token_to_id["<|endoftext|>".encode("utf-8")]
    return tok, eos_id


def load_model(device: str) -> TransformerLM:
    # Matches configs/train_tinystories.json (and checkpoint tensor shapes in this repo).
    return TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
    ).to(device)


def do_generate(
    *,
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
    device: str,
) -> tuple[list[int], list[int], str]:
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )
    out_ids = out[0].tolist()
    text = tokenizer.decode(out_ids)
    return prompt_ids, out_ids, text


def main() -> None:
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, eos_id = load_tokenizer(args.tokenizer)

    model = load_model(device)
    opt = AdamW(model.parameters())
    load_checkpoint(args.checkpoint, model, opt)
    model.eval()

    eos_token_id = eos_id if args.stop_on_eos else None

    prompt_ids, out_ids, text = do_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    new_tokens = len(out_ids) - len(prompt_ids)

    # Optional: ensure >=256 new tokens even if EOS triggers early.
    if args.force_at_least_256 and args.stop_on_eos and new_tokens < 256:
        prompt_ids, out_ids, text = do_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=max(args.max_new_tokens, 256),
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=None,
            device=device,
        )
        new_tokens = len(out_ids) - len(prompt_ids)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"tokenizer: {args.tokenizer}")
    print(
        f"temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens} "
        f"stop_on_eos={args.stop_on_eos} seed={args.seed}"
    )
    print(f"prompt_tokens: {len(prompt_ids)}")
    print(f"total_tokens: {len(out_ids)}")
    print(f"new_tokens: {new_tokens}")
    print("---TEXT_START---")
    print(text)
    print("---TEXT_END---")


if __name__ == "__main__":
    main()
