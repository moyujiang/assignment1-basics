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
import codecs
import json
import multiprocessing as mp
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator

import numpy as np

from cs336_basics.tokenization.bpe_tokenizer import BPETokenizer
from cs336_basics.tokenization.pretokenization_example import find_chunk_boundaries


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
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            s = decoder.decode(b)
            if s:
                yield s
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def iter_text_from_byte_range(path: Path, start: int, end: int, read_bytes: int) -> Iterator[str]:
    """Stream UTF-8 text decoded from a byte range [start, end).

    Uses an incremental UTF-8 decoder so we don't lose characters when read
    boundaries fall inside multibyte codepoints.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    remaining = end - start
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            b = f.read(min(read_bytes, remaining))
            if not b:
                break
            remaining -= len(b)
            s = decoder.decode(b)
            if s:
                yield s
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def count_tokens(tokenizer: BPETokenizer, text_chunks: Iterable[str]) -> int:
    return sum(1 for _ in tokenizer.encode_iterable(text_chunks))


_WORKER_TOKENIZER: BPETokenizer | None = None


def _init_worker_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def _count_range_worker(args: tuple[str, int, int, int]) -> int:
    """Count tokens in a file byte-range."""
    path_s, start, end, read_bytes = args
    assert _WORKER_TOKENIZER is not None
    path = Path(path_s)
    return sum(1 for _ in _WORKER_TOKENIZER.encode_iterable(iter_text_from_byte_range(path, start, end, read_bytes)))


def _write_range_worker(args: tuple[str, str, int, int, int, int, int]) -> int:
    """Write tokens for a file byte-range into a shared memmap slice.

    Args tuple:
      input_path, output_path, total_tokens, start, end, offset, read_bytes
    Returns: number of tokens written.
    """
    input_path_s, output_path_s, total_tokens, start, end, offset, read_bytes = args
    assert _WORKER_TOKENIZER is not None

    input_path = Path(input_path_s)
    output_path = Path(output_path_s)
    arr = np.lib.format.open_memmap(output_path, mode="r+", dtype=np.uint16, shape=(total_tokens,))

    idx = int(offset)
    written = 0
    buf: list[int] = []
    buf_append = buf.append

    for tok in _WORKER_TOKENIZER.encode_iterable(iter_text_from_byte_range(input_path, start, end, read_bytes)):
        buf_append(tok)
        if len(buf) >= 262_144:
            n = len(buf)
            arr[idx : idx + n] = np.asarray(buf, dtype=np.uint16)
            idx += n
            written += n
            buf.clear()

    if buf:
        n = len(buf)
        arr[idx : idx + n] = np.asarray(buf, dtype=np.uint16)
        idx += n
        written += n
        buf.clear()

    return written


def write_tokens_uint16(
    *,
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    chunk_bytes: int,
    block_tokens: int,
    overwrite: bool,
    workers: int,
    doc_sep: str,
) -> dict[str, float]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    max_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    if max_id >= 2**16:
        raise ValueError(f"Tokenizer has token id {max_id}, which does not fit in uint16")

    # Ensure doc_sep is representable as a special token and present in vocab.
    sep_bytes = doc_sep.encode("utf-8")
    if sep_bytes not in tokenizer.token_to_id:
        raise ValueError(
            f"doc-sep {doc_sep!r} is not in tokenizer vocab (did you pass it as a special token?)"
        )

    # Pass 1: count tokens
    t0 = perf_counter()
    if workers <= 1:
        n_tokens = count_tokens(tokenizer, iter_text_chunks(input_path, chunk_bytes))
    else:
        # Split the file into byte ranges aligned to the special token boundary.
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=workers,
                split_special_token=sep_bytes,
            )
        ranges = list(zip(boundaries[:-1], boundaries[1:]))

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker_tokenizer,
            initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
        ) as pool:
            counts = pool.map(
                _count_range_worker,
                [(str(input_path), start, end, chunk_bytes) for start, end in ranges],
            )
        n_tokens = int(sum(counts))
    t1 = perf_counter()

    # Pass 2: write tokens
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(n_tokens,))

    idx = 0
    buf: list[int] = []
    buf_extend = buf.extend

    def flush_buf() -> None:
        nonlocal idx
        if not buf:
            return
        n = len(buf)
        arr[idx : idx + n] = np.asarray(buf, dtype=np.uint16)
        idx += n
        buf.clear()

    t2 = perf_counter()
    if workers <= 1:
        for tok in tokenizer.encode_iterable(iter_text_chunks(input_path, chunk_bytes)):
            buf.append(tok)
            if len(buf) >= block_tokens:
                flush_buf()
        flush_buf()
    else:
        # Recompute ranges (small cost) to avoid plumbing through count phase.
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=workers,
                split_special_token=sep_bytes,
            )
        ranges = list(zip(boundaries[:-1], boundaries[1:]))

        # Count per range again to compute offsets (still cheap vs writing, and keeps code simple).
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker_tokenizer,
            initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
        ) as pool:
            counts = pool.map(
                _count_range_worker,
                [(str(input_path), start, end, chunk_bytes) for start, end in ranges],
            )

        offsets: list[int] = []
        running = 0
        for c in counts:
            offsets.append(running)
            running += int(c)
        if running != n_tokens:
            raise RuntimeError(f"Token count mismatch between passes: pass1={n_tokens}, pass2={running}")

        with ctx.Pool(
            processes=workers,
            initializer=_init_worker_tokenizer,
            initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
        ) as pool:
            written_counts = pool.map(
                _write_range_worker,
                [
                    (
                        str(input_path),
                        str(output_path),
                        int(n_tokens),
                        int(start),
                        int(end),
                        int(offsets[i]),
                        int(chunk_bytes),
                    )
                    for i, (start, end) in enumerate(ranges)
                ],
            )
        idx = int(sum(int(x) for x in written_counts))

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
        "count_MBps": (input_bytes / max(1e-9, (t1 - t0))) / 1e6,
        "write_MBps": (input_bytes / max(1e-9, (t3 - t2))) / 1e6,
        "tokens_per_second": float(n_tokens) / max(1e-9, (t3 - t2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode datasets to uint16 .npy token-id arrays")
    ap.add_argument("--which", choices=["tinystories", "owt", "all"], default="all")
    ap.add_argument("--splits", type=str, default="train,valid", help="Comma-separated: train,valid")
    ap.add_argument("--out-dir", type=str, default="data/tokenized")
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--block-tokens", type=int, default=1_000_000)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for tokenization (0 = auto from CPU/GPU counts)",
    )
    ap.add_argument("--doc-sep", type=str, default="<|endoftext|>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Auto workers: prefer CPU parallelism; GPU count is included only as a fallback signal.
    if args.workers <= 0:
        cpu_workers = os.cpu_count() or 1
        try:
            import torch

            gpu_workers = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            gpu_workers = 0

        args.workers = max(1, cpu_workers, gpu_workers)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_dir = root / args.out_dir

    special_tokens = [args.doc_sep]
    chunk_bytes = args.chunk_mb * (1 << 20)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in ("train", "valid"):
            raise ValueError(f"Unknown split: {s} (expected train/valid)")

    def run_one(name: str, tok_out_dir: Path, train_path: Path, valid_path: Path) -> None:
        tokenizer = load_tokenizer(
            vocab_path=tok_out_dir / "vocab.json",
            merges_path=tok_out_dir / "merges.txt",
            special_tokens=special_tokens,
        )

        split_to_path = {"train": train_path, "valid": valid_path}
        for split in splits:
            in_path = split_to_path[split]
            out_path = out_dir / f"{name}_{split}.uint16.npy"
            stats = write_tokens_uint16(
                tokenizer=tokenizer,
                input_path=in_path,
                output_path=out_path,
                chunk_bytes=chunk_bytes,
                block_tokens=args.block_tokens,
                overwrite=args.overwrite,
                workers=args.workers,
                doc_sep=args.doc_sep,
            )

            print(f"[{name}:{split}] -> {out_path}")
            print(
                f"  tokens={int(stats['n_tokens']):,}  "
                f"count={stats['count_seconds']:.2f}s ({stats['count_MBps']:.2f} MB/s)  "
                f"write={stats['write_seconds']:.2f}s ({stats['write_MBps']:.2f} MB/s)  "
                f"encode={stats['tokens_per_second']:.0f} tok/s"
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
