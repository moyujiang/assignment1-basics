"""
Train a byte-level BPE tokenizer on the TinyStories dataset.
"""

import os
import json
import time
import tracemalloc
from pathlib import Path
from cs336_basics.tokenization.bpe_tokenizer import BPETokenizer


def main():
    # Paths
    data_dir = Path("../data")
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    # input_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    output_dir = Path("./tokenizer_output_tinystories")
    output_dir.mkdir(exist_ok=True)

    # Configuration
    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on {input_path}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print("-" * 50)

    # Track memory usage
    tracemalloc.start()
    start_time = time.time()

    timings: dict[str, float] = {}

    # Train the tokenizer
    vocab, merges = BPETokenizer.train_bpe(
        input_path=str(input_path),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        timings=timings,
        verbose=True,
    )

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_time = end_time - start_time

    # Report results
    print("-" * 50)
    print(f"Training completed!")
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/3600:.4f} hours)")
    if timings:
        print(f"Pretokenize time: {timings.get('pretokenize_s', float('nan')):.2f} seconds")
        print(f"Train time: {timings.get('train_s', float('nan')):.2f} seconds")
    print(f"Peak memory usage: {peak / (1024**3):.2f} GB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Find longest token
    longest_token = max(vocab.values(), key=lambda x: len(x))
    longest_token_str = longest_token.decode("utf-8", errors="replace")
    print(f"\nLongest token: {longest_token!r} ({len(longest_token)} bytes)")
    print(f"As string: {longest_token_str!r}")

    # Serialize vocab and merges to disk
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    # Save vocab as JSON (convert bytes to list of ints for JSON serialization)
    vocab_serializable = {
        str(k): list(v) for k, v in vocab.items()
    }
    with open(vocab_path, "w") as f:
        json.dump(vocab_serializable, f)
    print(f"\nVocabulary saved to {vocab_path}")

    # Save merges as text
    with open(merges_path, "w") as f:
        for token1, token2 in merges:
            # Convert bytes to strings for serialization
            f.write(f"{token1!r} {token2!r}\n")
    print(f"Merges saved to {merges_path}")

    return vocab, merges, elapsed_time, peak


if __name__ == "__main__":
    main()
