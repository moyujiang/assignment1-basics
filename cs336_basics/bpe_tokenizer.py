"""
Implement BPE tokenizer
"""

import os
import regex as re
import multiprocessing as mp
import functools
import heapq
from typing import List, Tuple, Dict
from typing import Iterable, Iterator
from collections import Counter, defaultdict
from .pretokenization_example import find_chunk_boundaries

# GPT-2 pretokenization regex pattern
PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        vocab: Dict[int, bytes], mapping from token ID to byte string
        merges: List[Tuple[bytes, bytes]], list of byte pair merges
        special_tokens: list[str] | None, list of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Hot-path helpers
        self._special_tokens_set = set(self.special_tokens)
        # Precompute single-byte tokens to avoid per-byte allocations in encode()
        self._byte_tokens = [bytes([i]) for i in range(256)]
        # Cache: pretoken bytes -> list[int] token ids
        self._encode_cache: Dict[bytes, List[int]] = {}
        self._encode_cache_max_entries = 100_000

        # reverse map 
        self.token_to_id = {v: k for k, v in vocab.items()}

        # merge ranks
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # compile pretokenization pattern
        self._pre_pat = re.compile(PATTERN)

        # compile special-token matcher (used to split BEFORE pretokenization)
        specials = sorted(self.special_tokens, key=len, reverse=True)
        if specials:
            split_pattern = "|".join(map(re.escape, specials))
            self._special_split_re = re.compile(f"({split_pattern})")
        else:
            self._special_split_re = None

    # encode / decode

    def encode(self, text: str) -> List[int]:
        """
        Encode input text to list of token IDs
        """
        ids: List[int] = []

        # Important: split on special tokens BEFORE pretokenization.
        # Otherwise the GPT-2 pretokenization regex (which includes an optional leading space)
        # can swallow a special token (e.g., matching " <|endoftext|>" as one piece),
        # preventing us from recognizing it.
        parts: List[str]
        if self._special_split_re is not None:
            parts = [p for p in self._special_split_re.split(text) if p]
        else:
            parts = [text]

        for part in parts:
            if self._special_tokens_set and part in self._special_tokens_set:
                b = part.encode("utf-8")
                ids.append(self.token_to_id[b])
                continue

            for match in self._pre_pat.finditer(part):
                piece = match.group(0)
                b = piece.encode("utf-8")

                cached = self._encode_cache.get(b)
                if cached is not None:
                    ids.extend(cached)
                    continue

                symbols = [self._byte_tokens[x] for x in b]
                symbols = self._bpe(symbols)
                out_ids = [self.token_to_id[sym] for sym in symbols]
                ids.extend(out_ids)

                # Simple bounded cache (high hit rate on natural language)
                if len(self._encode_cache) >= self._encode_cache_max_entries:
                    self._encode_cache.clear()
                self._encode_cache[b] = out_ids
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode list of token IDs back to string
        """
        bs = b"".join([self.vocab[i] for i in ids])
        # Individual tokens may not align to UTF-8 character boundaries (e.g., a token could be b"\xc3").
        # Tests decode per-token for inspection, so avoid raising here.
        return bs.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        guard = max(0, max_special_len - 1)

        def safe_emittable_end(text: str) -> int:
            """Return a safe cut position (< len(text)) aligned to tokenization units.

            We avoid emitting the final unit that touches the end of `text`, since that
            unit could change if more characters arrive (e.g., a partial word, trailing
            whitespace that becomes a leading-space token, or a partial special token).
            """
            n = len(text)
            if n == 0:
                return 0

            # Keep a tail region un-emitted so that we never split a special token
            # (or any other tokenization unit) across chunk boundaries.
            emit_limit = n - guard if guard > 0 else n - 1
            if emit_limit <= 0:
                return 0

            safe_end = 0
            if self._special_split_re is None:
                for m in self._pre_pat.finditer(text):
                    end = m.end()
                    if end <= emit_limit:
                        safe_end = end
                return safe_end

            pos = 0
            for part in self._special_split_re.split(text):
                if part == "":
                    continue
                part_end = pos + len(part)
                if part in self.special_tokens:
                    if part_end <= emit_limit:
                        safe_end = part_end
                else:
                    for m in self._pre_pat.finditer(part):
                        end = pos + m.end()
                        if end <= emit_limit:
                            safe_end = end
                pos = part_end
            return safe_end

        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk

            while True:
                safe_end = safe_emittable_end(buffer)
                if safe_end <= 0:
                    break
                yield from self.encode(buffer[:safe_end])
                buffer = buffer[safe_end:]

        if buffer:
            yield from self.encode(buffer)

    # Internal BPE
    def _bpe(self, symbols: List[bytes]) -> List[bytes]:
        """
        Apply BPE to list of byte symbols
        """
        if len(symbols) == 1:
            return symbols

        while True:
            # find best merge
            best_rank = None
            best_pair = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break
            
            # merge all occurrences of best_pair
            new_symbols: List[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        return symbols

    @staticmethod
    def _count_chunk(
        chunk_bytes: bytes,
        special_tokens: List[str],
    ) -> Counter:
        """
        Count pre-tokens within a single chunk (bytes).
        First splits on special tokens, then pre-tokenizes the non-special parts.
        Returns Counter mapping token tuple to frequency.
        """
        text = chunk_bytes.decode("utf-8", errors="ignore")
        
        # First: split on special tokens to isolate them
        if special_tokens:
            split_pattern = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
            parts = re.split(f"({split_pattern})", text)
        else:
            parts = [text]
        
        # Second: pretokenize each part and count
        counter = Counter()
        pat = re.compile(PATTERN)
        
        for part in parts:
            if not part:
                continue
            # Check if this part is a special token
            if part in special_tokens:
                word = (part.encode("utf-8"),)
                counter[word] += 1
            else:
                # Pretokenize with regex
                for match in pat.finditer(part):
                    piece = match.group(0)
                    b = piece.encode("utf-8")
                    word = tuple(bytes([x]) for x in b)
                    counter[word] += 1
        
        return counter


    # Training BPE

    @staticmethod
    def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
        timings: Dict[str, float] | None = None,
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Train BPE on a corpus, returning vocab (id -> bytes) and merges.
        Chunks on special tokens for parallel pre-tokenization counting,
        then performs sequential BPE merges.
        """

        from time import perf_counter

        t_pretok0 = perf_counter()

        # Chunk the binary file using find_chunk_boundaries
        split_tok = special_tokens[0].encode("utf-8") if special_tokens else b""
        with open(input_path, "rb") as f:
            if split_tok:
                boundaries = find_chunk_boundaries(f, desired_num_chunks=os.cpu_count() or 4, split_special_token=split_tok)
            else:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                boundaries = [0, file_size]
            
            chunks: List[bytes] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start))

        # Parallel pre-tokenization counting per chunk
        word_freq = Counter()
        with mp.Pool(processes=os.cpu_count() or 4) as pool:
            for chunk_counter in pool.imap_unordered(
                functools.partial(BPETokenizer._count_chunk, special_tokens=special_tokens or []),
                chunks,
                chunksize=1
            ):
                word_freq.update(chunk_counter)

        if timings is not None:
            timings["pretokenize_s"] = perf_counter() - t_pretok0

        t_train0 = perf_counter()

        # initialize vocab with single bytes + special tokens
        vocab: Dict[int, bytes] = {}
        for b in range(256):
            vocab[len(vocab)] = bytes([b])
        for sp in special_tokens or []:
            b = sp.encode("utf-8")
            if b not in vocab.values():
                vocab[len(vocab)] = b
        
        merges: List[Tuple[bytes, bytes]] = []

        # Assign ID to each unique word and maintain mutable token sequences
        word_tokens: Dict[int, List[bytes]] = {}
        word_counts: Dict[int, int] = {}

        word_id = 0
        for word, freq in word_freq.items():
            word_tokens[word_id] = list(word)
            word_counts[word_id] = freq
            word_id += 1

        def _pair_counts(tokens: List[bytes]) -> Counter:
            c = Counter()
            for a, b in zip(tokens, tokens[1:]):
                c[(a, b)] += 1
            return c

        def _rev_bytes_key(bs: bytes) -> Tuple[int, ...]:
            # Reverse lexicographic order for bytes, including prefix cases.
            # If x is a prefix of y, then x < y in Python bytes ordering, so in
            # reverse order x should be > y. Appending a sentinel > 255 achieves that.
            return tuple((255 - b) for b in bs) + (256,)

        def _heap_key(
            pair: Tuple[bytes, bytes],
            freq: int,
        ) -> Tuple[int, Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[bytes, bytes]]:
            # We want: max freq; tie-break: lexicographically largest pair on original bytes.
            # heapq pops smallest, so use (-freq) and a reversed-bytes key.
            a, b = pair
            return (-freq, (_rev_bytes_key(a), _rev_bytes_key(b)), pair)

        def _merge_tokens(tokens: List[bytes], pair: Tuple[bytes, bytes], merged: bytes) -> List[bytes]:
            if len(tokens) < 2:
                return tokens
            out: List[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            return out

        # Build global pair frequency, plus an index from pair -> affected word_ids.
        pair_freq: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[bytes, bytes], set[int]] = defaultdict(set)
        word_pair_counts: Dict[int, Counter] = {}

        for wid, tokens in word_tokens.items():
            wc = word_counts[wid]
            pc = _pair_counts(tokens)
            word_pair_counts[wid] = pc
            for pair, cnt in pc.items():
                pair_freq[pair] += cnt * wc
                pair_to_words[pair].add(wid)

        # Heap with lazy deletion for best-pair selection.
        heap: List[Tuple[int, Tuple[bytes, bytes], Tuple[bytes, bytes]]] = []
        for pair, freq in pair_freq.items():
            if freq > 0:
                heapq.heappush(heap, _heap_key(pair, freq))

        # Iteratively merge most frequent pairs.
        while len(vocab) < vocab_size and heap:
            print(f"Vocab size: {len(vocab)}; Heap size: {len(heap)}", end="\r")
            best_pair: Tuple[bytes, bytes] | None = None
            best_freq = 0

            while heap:
                neg_f, _rev, candidate = heapq.heappop(heap)
                freq = pair_freq.get(candidate, 0)
                if freq == 0:
                    continue
                if freq == -neg_f:
                    best_pair = candidate
                    best_freq = freq
                    break

            if best_pair is None or best_freq == 0:
                break

            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]

            affected_wids = list(pair_to_words.get(best_pair, set()))
            touched_pairs: set[Tuple[bytes, bytes]] = set()

            for wid in affected_wids:
                old_tokens = word_tokens[wid]
                new_tokens = _merge_tokens(old_tokens, best_pair, merged_token)
                if new_tokens == old_tokens:
                    continue

                wc = word_counts[wid]
                old_pc = word_pair_counts[wid]
                new_pc = _pair_counts(new_tokens)

                # Remove old contributions
                for pair, cnt in old_pc.items():
                    pair_freq[pair] -= cnt * wc
                    pair_to_words[pair].discard(wid)
                    touched_pairs.add(pair)

                # Add new contributions
                for pair, cnt in new_pc.items():
                    pair_freq[pair] += cnt * wc
                    pair_to_words[pair].add(wid)
                    touched_pairs.add(pair)

                word_tokens[wid] = new_tokens
                word_pair_counts[wid] = new_pc

            # Push updated heap keys for touched pairs (lazy deletion handles stale keys).
            for pair in touched_pairs:
                freq = pair_freq.get(pair, 0)
                if freq > 0:
                    heapq.heappush(heap, _heap_key(pair, freq))

            # Add merged token to vocab
            if merged_token not in set(vocab.values()):
                vocab[len(vocab)] = merged_token

        if timings is not None:
            timings["train_s"] = perf_counter() - t_train0

        return vocab, merges

    @classmethod
    def from_train(cls, input_path: str, vocab_size: int, special_tokens: List[str] | None = None) -> "BPETokenizer":
        vocab, merges = cls.train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens: List[str] | None = None) -> "BPETokenizer":
        """
        Load BPE tokenizer from vocab and merges files.
        vocab_filepath: path to vocab.json
        merge_filepath: path to merges.txt
        special_tokens: list of special tokens
        """
        # Load vocab
        with open(vocab_filepath, "r") as f:
            vocab_serializable = json.load(f)
        vocab: Dict[int, bytes] = {
            int(k): bytes(v) for k, v in vocab_serializable.items()
        }

        # Load merges
        merges: List[Tuple[bytes, bytes]] = []
        with open(merge_filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)