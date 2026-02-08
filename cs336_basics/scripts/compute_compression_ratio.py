"""
Sample documents from TinyStories and OpenWebText, encode with trained tokenizers,
report compression ratio (bytes/token).
"""

from __future__ import annotations

import ast
import argparse
import json
import random
import re
from time import perf_counter
from pathlib import Path
from typing import Iterator, Iterable

from cs336_basics.tokenization.bpe_tokenizer import BPETokenizer


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


def iter_documents(path: Path, sep: str, chunk_size: int = 1 << 20) -> Iterator[str]:
    """Yield documents split by a separator token, streaming from disk."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(sep)
            for doc in parts[:-1]:
                yield doc
            buffer = parts[-1]
        if buffer:
            yield buffer


def reservoir_sample(iterable: Iterable[str], k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def compression_ratio_bytes_per_token(tokenizer: BPETokenizer, docs: list[str]) -> float:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        doc_bytes = doc.encode("utf-8")
        total_bytes += len(doc_bytes)
        total_tokens += len(tokenizer.encode(doc))
    return total_bytes / max(1, total_tokens)


def total_tokens(tokenizer: BPETokenizer, docs: list[str]) -> int:
    return sum(len(tokenizer.encode(doc)) for doc in docs)


def measure_encode_throughput(tokenizer: BPETokenizer, docs: list[str]) -> dict[str, float]:
    """Measure encode throughput on a fixed set of docs.

    Returns a dict with:
      - total_bytes
      - total_tokens
      - seconds
      - bytes_per_token
      - bytes_per_second
      - tokens_per_second
    """
    total_bytes = 0
    total_tokens = 0

    t0 = perf_counter()
    for doc in docs:
        doc_bytes = doc.encode("utf-8")
        total_bytes += len(doc_bytes)
        total_tokens += len(tokenizer.encode(doc))
    t1 = perf_counter()

    seconds = max(1e-9, t1 - t0)
    bytes_per_token = total_bytes / max(1, total_tokens)
    bytes_per_second = total_bytes / seconds
    tokens_per_second = total_tokens / seconds
    return {
        "total_bytes": float(total_bytes),
        "total_tokens": float(total_tokens),
        "seconds": float(seconds),
        "bytes_per_token": float(bytes_per_token),
        "bytes_per_second": float(bytes_per_second),
        "tokens_per_second": float(tokens_per_second),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute bytes/token compression and encode throughput on sampled docs.")
    ap.add_argument("--k", type=int, default=10, help="Number of documents to sample per dataset")
    ap.add_argument("--seed", type=int, default=42, help="Reservoir sampling seed")
    ap.add_argument("--ts-split", choices=["train", "valid"], default="train")
    ap.add_argument("--owt-split", choices=["train", "valid"], default="train")
    ap.add_argument("--doc-sep", type=str, default="<|endoftext|>")
    ap.add_argument("--chunk-mb", type=int, default=1, help="Read chunk size for document iteration (MB)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    # Tokenizer outputs
    ts_output = root / "cs336_basics" / "tokenizer_output_tinystories"
    owt_output = root / "cs336_basics" / "tokenizer_output_owt"

    ts_tokenizer = load_tokenizer(
        vocab_path=ts_output / "vocab.json",
        merges_path=ts_output / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    owt_tokenizer = load_tokenizer(
        vocab_path=owt_output / "vocab.json",
        merges_path=owt_output / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    # Sample documents
    ts_path = data_dir / f"TinyStoriesV2-GPT4-{args.ts_split}.txt"
    owt_path = data_dir / f"owt_{args.owt_split}.txt"
    chunk_size = args.chunk_mb * (1 << 20)

    ts_docs = reservoir_sample(
        iter_documents(ts_path, args.doc_sep, chunk_size=chunk_size),
        k=args.k,
        seed=args.seed,
    )
    owt_docs = reservoir_sample(
        iter_documents(owt_path, args.doc_sep, chunk_size=chunk_size),
        k=args.k,
        seed=args.seed,
    )

    ts_stats = measure_encode_throughput(ts_tokenizer, ts_docs)
    owt_stats = measure_encode_throughput(owt_tokenizer, owt_docs)

    # Cross-tokenize: OWT docs encoded with TinyStories tokenizer
    owt_with_ts_stats = measure_encode_throughput(ts_tokenizer, owt_docs)

    print(
        "TinyStories tokenizer (10K) compression ratio (bytes/token):",
        f"{ts_stats['bytes_per_token']:.4f}",
    )
    print(
        "TinyStories tokenizer throughput:",
        f"{ts_stats['bytes_per_second'] / 1e6:.2f} MB/s",
        f"({ts_stats['tokens_per_second']:.0f} tok/s)",
    )

    print(
        "OpenWebText tokenizer (32K) compression ratio (bytes/token):",
        f"{owt_stats['bytes_per_token']:.4f}",
    )
    print(
        "OpenWebText tokenizer throughput:",
        f"{owt_stats['bytes_per_second'] / 1e6:.2f} MB/s",
        f"({owt_stats['tokens_per_second']:.0f} tok/s)",
    )

    print(
        "OpenWebText sample with TinyStories tokenizer (bytes/token):",
        f"{owt_with_ts_stats['bytes_per_token']:.4f}",
    )
    print(
        "TinyStories tokenizer throughput on OWT:",
        f"{owt_with_ts_stats['bytes_per_second'] / 1e6:.2f} MB/s",
        f"({owt_with_ts_stats['tokens_per_second']:.0f} tok/s)",
    )

    if owt_stats["total_tokens"] > 0 and owt_with_ts_stats["total_tokens"] > 0:
        token_multiplier = owt_with_ts_stats["total_tokens"] / owt_stats["total_tokens"]
        ratio_delta_pct = (
            100.0 * (owt_with_ts_stats["bytes_per_token"] - owt_stats["bytes_per_token"]) / owt_stats["bytes_per_token"]
            if owt_stats["bytes_per_token"] > 0
            else float("nan")
        )
        print(
            "OWT token count multiplier (TinyStories vs OWT tokenizer):",
            f"{token_multiplier:.3f}x",
        )
        print("OWT bytes/token change vs OWT tokenizer:", f"{ratio_delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
