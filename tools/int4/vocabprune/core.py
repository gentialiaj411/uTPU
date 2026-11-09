"""Deterministic llama2.c byte-fallback vocabulary pruning primitives."""
from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.llama2c.tokenizer import Tokenizer

BASE_IDS = tuple(range(259))
KEEP_SIZES = (2048, 4096, 6144, 8192)
SEPARATOR = b"<|endoftext|>"


def fast_encode(tokenizer: Tokenizer, story: str) -> list[int]:
    """Encode one story with the tokenizer's exact merge rule, without quadratic scans."""
    if not story:
        return []
    tokens = [tokenizer.lookup[b" "]]
    for char in story:
        piece = char.encode("utf-8")
        token = tokenizer.lookup.get(piece)
        tokens.extend([b + 3 for b in piece] if token is None else [token])
    n = len(tokens)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    alive = [True] * n
    heap: list[tuple[float, int, int]] = []
    for i in range(n - 1):
        merged = tokenizer.lookup.get(tokenizer.vocab[tokens[i]] + tokenizer.vocab[tokens[i + 1]])
        if merged is not None:
            heapq.heappush(heap, (-tokenizer.scores[merged], i, merged))
    while heap:
        _, i, merged = heapq.heappop(heap)
        j = nxt[i]
        if j < 0 or not alive[i] or not alive[j]:
            continue
        if tokenizer.lookup.get(tokenizer.vocab[tokens[i]] + tokenizer.vocab[tokens[j]]) != merged:
            continue
        tokens[i] = merged
        alive[j] = False
        nxt[i] = nxt[j]
        if nxt[j] >= 0:
            prv[nxt[j]] = i
        for a in (prv[i], i):
            if a >= 0 and alive[a] and nxt[a] >= 0:
                z = tokenizer.lookup.get(tokenizer.vocab[tokens[a]] + tokenizer.vocab[tokens[nxt[a]]])
                if z is not None:
                    heapq.heappush(heap, (-tokenizer.scores[z], a, z))
    return [tokens[i] for i in range(n) if alive[i]]


def iter_stories(path: Path, chunk_bytes: int = 1 << 20):
    """Yield story strings from a separator-delimited UTF-8 file without loading it."""
    pending = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            pending += block
            parts = pending.split(SEPARATOR)
            pending = parts.pop()
            for part in parts:
                if part:
                    yield part.decode("utf-8")
    if pending:
        yield pending.decode("utf-8")


def count_tokens(tokenizer: Tokenizer, path: Path) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for story in iter_stories(path):
        ids = fast_encode(tokenizer, story)
        counts.update(ids)
        total += len(ids)
    return counts, total


def build_idmap(tokenizer: Tokenizer, counts: Counter, total: int, keep_size: int,
                sample_path: Path) -> dict:
    # RTL classifier tiles are 16 rows wide.  Keep the study presets above,
    # but allow constrained small-model reproducers to request any legal tile
    # count while retaining the complete byte-fallback prefix.
    if keep_size < len(BASE_IDS) or keep_size % 16 or keep_size > len(tokenizer.vocab):
        raise ValueError(f"keep_size must be a multiple of 16 in [{len(BASE_IDS)}, {len(tokenizer.vocab)}]")
    ranked = sorted(((int(token), int(counts[token])) for token in counts if token not in BASE_IDS),
                    key=lambda item: (-item[1], item[0]))
    kept = list(BASE_IDS) + [token for token, _ in ranked[:keep_size - len(BASE_IDS)]]
    if len(kept) != keep_size:
        raise ValueError(f"sample has only {len(kept)} distinct/base-covered rows for K={keep_size}")
    inverse = [-1] * len(tokenizer.vocab)
    for compact, original in enumerate(kept):
        inverse[original] = compact
    return {
        "schema_version": 1,
        "original_vocab": len(tokenizer.vocab),
        "keep_size": keep_size,
        "kept_original_ids": kept,
        "original_to_compact": inverse,
        "special_ids": [0, 1, 2],
        "byte_fallback_ids": list(range(3, 259)),
        "ordering": "specials 0..2, byte fallback 3..258, then (-frequency, original_id)",
        "frequency_sample": str(sample_path),
        "frequency_sample_bytes": sample_path.stat().st_size,
        "frequency_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "models/llama2c/tokenizer.bin").read_bytes()).hexdigest(),
        "sample_token_occurrences": total,
        "ranked_rows": [{"original_id": token, "count": count} for token, count in ranked[:keep_size - len(BASE_IDS)]],
    }


def encode_with_fallback(tokenizer: Tokenizer, text: str, idmap: dict) -> tuple[list[int], list[int], int, int]:
    """Return compact IDs, corresponding original IDs, fallback count, original token count."""
    inverse = np.asarray(idmap["original_to_compact"], dtype=np.int64)
    compact: list[int] = []
    expanded_original: list[int] = []
    fallback = 0
    original = fast_encode(tokenizer, text)
    for token in original:
        compact_id = int(inverse[token])
        if compact_id >= 0:
            compact.append(compact_id)
            expanded_original.append(token)
            continue
        fallback += 1
        piece = tokenizer.vocab[token]
        for byte in piece:
            byte_token = int(byte) + 3
            byte_compact = int(inverse[byte_token])
            if byte_compact < 0:
                raise ValueError(f"byte fallback row {byte_token} was not kept")
            compact.append(byte_compact)
            expanded_original.append(byte_token)
    return compact, expanded_original, fallback, len(original)


def decode_compact(tokenizer: Tokenizer, compact_ids: list[int], idmap: dict) -> str:
    original = [int(idmap["kept_original_ids"][x]) for x in compact_ids]
    return tokenizer.decode(original)


def prune_export(export: dict, kept_original_ids: list[int]) -> dict:
    """Return a copy of a tied numerics_v2 export with only kept rows."""
    ids = np.asarray(kept_original_ids, dtype=np.int64)
    out = dict(export)
    out["config"] = dict(export["config"])
    out["config"]["vocab_size"] = int(ids.size)
    out["embedding"] = np.asarray(export["embedding"])[ids].copy()
    classifier = dict(export["classifier"])
    for key in ("weight", "multiplier", "shift", "weight_scale_multiplier", "weight_scale_shift",
                "embedding_scale_multiplier", "embedding_scale_shift"):
        if key in classifier:
            classifier[key] = np.asarray(classifier[key])[ids].copy()
    out["classifier"] = classifier
    if "embedding_group_multiplier" in export:
        out["embedding_group_multiplier"] = np.asarray(export["embedding_group_multiplier"])[ids].copy()
        out["embedding_group_shift"] = np.asarray(export["embedding_group_shift"])[ids].copy()
    if "embedding_scale_multiplier" in export:
        out["embedding_scale_multiplier"] = np.asarray(export["embedding_scale_multiplier"])[ids].copy()
    return out


def write_idmap(path: Path, idmap: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idmap, indent=2) + "\n", encoding="utf-8")
