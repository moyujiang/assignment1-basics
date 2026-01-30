"""Encode datasets into token-id arrays.

Task (assignment prompt):
- Using your TinyStories (10K) and OpenWebText (32K) tokenizers, encode the respective
  training and development datasets into a sequence of integer token IDs.
- Serialize token IDs as NumPy arrays of dtype uint16.

This script does a 2-pass streaming encode to avoid holding all token IDs in memory:
1) Count tokens
2) Write a .npy via numpy.lib.format.open_memmap

Default inputs assume the repo layout from the assignment README.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator

import numpy as np

from cs336_basics.bpe_tokenizer import BPETokenizer


BYTES_LITERAL_RE = re.compile(r"b'(?:\\.|[^'])*'|b\"(?:\\.|[^\"])*\"")


def load_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> BPETokenizer:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_serializable = json.load(f)
    vocab = {int(k): bytes(v) for k, v in vocab_serializable.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            matches = BYTES_LITERAL_RE.findall(line)
            if len(matches) != 2:
                continue
            a = ast.literal_eval(matches[0])
            b = ast.literal_eval(matches[1])
            merges.append((a, b))

    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def iter_text_chunks(path: Path, chunk_bytes: int) -> Iterator[str]:
    # Text mode is fine here because BPETokenizer.encode_iterable is designed to
    # handle chunk boundaries deterministically via buffering.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            s = f.read(chunk_bytes)
            if not s:
                break
            yield s


def count_tokens(tokenizer: BPETokenizer, text_chunks: Iterable[str]) -> int:
    return sum(1 for _ in tokenizer.encode_iterable(text_chunks))


def write_tokens_uint16(
    *,
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    chunk_bytes: int,
    block_tokens: int,
    overwrite: bool,
) -> dict[str, float]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    max_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    if max_id >= 2**16:
        raise ValueError(f"Tokenizer has token id {max_id}, which does not fit in uint16")

    # Pass 1: count tokens
    t0 = perf_counter()
    n_tokens = count_tokens(tokenizer, iter_text_chunks(input_path, chunk_bytes))
    t1 = perf_counter()

    # Pass 2: write tokens
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(n_tokens,))

    idx = 0
    buf: list[int] = []
    buf_append = buf.append

    t2 = perf_counter()
    for tok in tokenizer.encode_iterable(iter_text_chunks(input_path, chunk_bytes)):
        buf_append(tok)
        if len(buf) >= block_tokens:
            n = len(buf)
            arr[idx : idx + n] = np.asarray(buf, dtype=np.uint16)
            idx += n
            buf.clear()

    if buf:
        n = len(buf)
        arr[idx : idx + n] = np.asarray(buf, dtype=np.uint16)
        idx += n
        buf.clear()

    if idx != n_tokens:
        raise RuntimeError(f"Token count mismatch: counted {n_tokens}, wrote {idx}")

    arr.flush()
    t3 = perf_counter()

    input_bytes = float(input_path.stat().st_size)
    return {
        "input_bytes": input_bytes,
        "n_tokens": float(n_tokens),
        "count_seconds": float(t1 - t0),
        "write_seconds": float(t3 - t2),
        "total_seconds": float((t1 - t0) + (t3 - t2)),
        "count_MBps": (input_bytes / max(1e-9, (t1 - t0))) / 1e6,
        "write_MBps": (input_bytes / max(1e-9, (t3 - t2))) / 1e6,
        "tokens_per_second": float(n_tokens) / max(1e-9, (t3 - t2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode TinyStories/OWT datasets to uint16 .npy token-id arrays")
    ap.add_argument("--which", choices=["tinystories", "owt", "all"], default="all")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--out-dir", type=str, default="data/tokenized")
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--block-tokens", type=int, default=1_000_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--special-token", type=str, default="<|endoftext|>")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir

    special_tokens = [args.special_token] if args.special_token else []
    chunk_bytes = args.chunk_mb * (1 << 20)

    def run_one(name: str, tok_out_dir: Path, train_path: Path, valid_path: Path) -> None:
        tokenizer = load_tokenizer(
            vocab_path=tok_out_dir / "vocab.json",
            merges_path=tok_out_dir / "merges.txt",
            special_tokens=special_tokens,
        )

        for split, in_path in [("train", train_path), ("valid", valid_path)]:
            out_path = out_dir / f"{name}_{split}_uint16.npy"
            stats = write_tokens_uint16(
                tokenizer=tokenizer,
                input_path=in_path,
                output_path=out_path,
                chunk_bytes=chunk_bytes,
                block_tokens=args.block_tokens,
                overwrite=args.overwrite,
            )

            meta_path = out_dir / f"{name}_{split}_meta.json"
            meta = {
                "tokenizer": name,
                "split": split,
                "input_path": str(in_path),
                "output_path": str(out_path),
                "dtype": "uint16",
                "vocab_size": int(len(tokenizer.vocab)),
                "max_token_id": int(max(tokenizer.vocab.keys()) if tokenizer.vocab else 0),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"[{name}:{split}] wrote {int(stats['n_tokens']):,} tokens -> {out_path}")
            print(
                f"  count: {stats['count_seconds']:.2f}s ({stats['count_MBps']:.2f} MB/s), "
                f"write: {stats['write_seconds']:.2f}s ({stats['write_MBps']:.2f} MB/s), "
                f"encode: {stats['tokens_per_second']:.0f} tok/s"
            )

    if args.which in ("tinystories", "all"):
        run_one(
            "tinystories",
            tok_out_dir=root / "cs336_basics" / "tokenizer_output_tinystories",
            train_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            valid_path=data_dir / "TinyStoriesV2-GPT4-valid.txt",
        )

    if args.which in ("owt", "all"):
        run_one(
            "owt",
            tok_out_dir=root / "cs336_basics" / "tokenizer_output_owt",
            train_path=data_dir / "owt_train.txt",
            valid_path=data_dir / "owt_valid.txt",
        )


if __name__ == "__main__":
    # Avoid OpenMP oversubscription if the environment has it enabled.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
