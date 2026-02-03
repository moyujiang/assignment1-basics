"""
Train a byte-level BPE tokenizer on the OpenWebText dataset.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from cs336_basics.bpe_tokenizer import BPETokenizer


def main() -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], float, int]:
    # Paths
    data_dir = Path("../data")
    input_path = data_dir / "owt_train.txt"
    # input_path = data_dir / "owt_valid.txt"

    output_dir = Path("./tokenizer_output_owt")
    output_dir.mkdir(exist_ok=True)

    # Configuration
    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print("-" * 50)

    # Track memory usage
    tracemalloc.start()
    start_time = time.time()

    timings: dict[str, float] = {}

    vocab, merges = BPETokenizer.train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        timings=timings,
        verbose=True,
    )

    end_time = time.time()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_time = end_time - start_time

    # Report results
    print("-" * 50)
    print("Training completed!")
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/3600:.4f} hours)")
    if timings:
        print(f"Pretokenize time: {timings.get('pretokenize_s', float('nan')):.2f} seconds")
        print(f"Train time: {timings.get('train_s', float('nan')):.2f} seconds")
    print(f"Peak memory usage: {peak / (1024**3):.2f} GB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_str = longest_token.decode("utf-8", errors="replace")
    print(f"\nLongest token: {longest_token!r} ({len(longest_token)} bytes)")
    print(f"As string: {longest_token_str!r}")

    # Serialize vocab and merges to disk
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    vocab_serializable = {str(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(vocab_serializable, f)
    print(f"\nVocabulary saved to {vocab_path}")

    with open(merges_path, "w") as f:
        for token1, token2 in merges:
            f.write(f"{token1!r} {token2!r}\n")
    print(f"Merges saved to {merges_path}")

    return vocab, merges, elapsed_time, peak


if __name__ == "__main__":
    main()
