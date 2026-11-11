"""Float32 NumPy single-token decoder for llama2.c v0 checkpoints."""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_size * self.n_kv_heads


def _rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Force float32 reductions: NumPy otherwise preserves fp32 today, but this is contractual.
    ss = np.mean(x * x, dtype=np.float32) + np.float32(1e-5)
    return (weight * (x * np.float32(1.0 / np.sqrt(ss)))).astype(np.float32)


class Llama2CModel:
    """Memory-mapped v0 checkpoint with a run.c-shaped decode API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()[:28]
        if len(raw) != 28:
            raise ValueError("truncated llama2.c header")
        dim, hidden, layers, heads, kv_heads, signed_vocab, seq_len = struct.unpack("<7i", raw)
        shared = signed_vocab > 0
        self.config = ModelConfig(dim, hidden, layers, heads, kv_heads, abs(signed_vocab), seq_len, shared)
        c = self.config
        if min(dim, hidden, layers, heads, kv_heads, c.vocab_size, seq_len) <= 0 or dim % heads:
            raise ValueError(f"invalid llama2.c config: {c}")
        data = np.memmap(self.path, mode="r", dtype="<f4", offset=28)
        at = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal at
            count = int(np.prod(shape))
            if at + count > data.size:
                raise ValueError(f"truncated checkpoint while reading {shape}")
            out = data[at : at + count].reshape(shape)
            at += count
            return out

        self.token_embedding = take((c.vocab_size, c.dim))
        self.rms_att = take((c.n_layers, c.dim))
        self.wq = take((c.n_layers, c.dim, c.dim))
        self.wk = take((c.n_layers, c.kv_dim, c.dim))
        self.wv = take((c.n_layers, c.kv_dim, c.dim))
        self.wo = take((c.n_layers, c.dim, c.dim))
        self.rms_ffn = take((c.n_layers, c.dim))
        self.w1 = take((c.n_layers, c.hidden_dim, c.dim))
        self.w2 = take((c.n_layers, c.dim, c.hidden_dim))
        self.w3 = take((c.n_layers, c.hidden_dim, c.dim))
        self.rms_final = take((c.dim,))
        take((c.seq_len, c.head_size // 2))  # legacy RoPE table: real
        take((c.seq_len, c.head_size // 2))  # legacy RoPE table: imag
        self.wcls = self.token_embedding if shared else take((c.vocab_size, c.dim))
        if at != data.size:
            raise ValueError(f"checkpoint has {data.size - at} trailing float32 values")
        self.key_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), dtype=np.float32)
        self.value_cache = np.zeros_like(self.key_cache)

    def reset(self) -> None:
        self.key_cache.fill(0)
        self.value_cache.fill(0)

    def forward(self, token: int, pos: int) -> np.ndarray:
        """Return float32 logits for one token and update the KV cache at ``pos``."""
        c = self.config
        if not (0 <= token < c.vocab_size and 0 <= pos < c.seq_len):
            raise ValueError((token, pos))
        x = np.array(self.token_embedding[token], dtype=np.float32, copy=True)
        hs, kv_mul = c.head_size, c.n_heads // c.n_kv_heads
        for layer in range(c.n_layers):
            xb = _rmsnorm(x, self.rms_att[layer])
            q = (self.wq[layer] @ xb).astype(np.float32)
            k = (self.wk[layer] @ xb).astype(np.float32)
            v = (self.wv[layer] @ xb).astype(np.float32)
            for i in range(0, c.dim, 2):
                head_dim = i % hs
                freq = np.float32(1.0 / (10000.0 ** (head_dim / hs)))
                angle = np.float32(pos) * freq
                cs, sn = np.float32(np.cos(angle)), np.float32(np.sin(angle))
                q0, q1 = q[i], q[i + 1]
                q[i], q[i + 1] = q0 * cs - q1 * sn, q0 * sn + q1 * cs
                if i < c.kv_dim:
                    k0, k1 = k[i], k[i + 1]
                    k[i], k[i + 1] = k0 * cs - k1 * sn, k0 * sn + k1 * cs
            self.key_cache[layer, pos] = k
            self.value_cache[layer, pos] = v
            heads = np.empty((c.n_heads, hs), dtype=np.float32)
            scale = np.float32(1.0 / np.sqrt(hs))
            for h in range(c.n_heads):
                qh = q[h * hs : (h + 1) * hs]
                kh = self.key_cache[layer, : pos + 1, (h // kv_mul) * hs : (h // kv_mul + 1) * hs]
                scores = (kh @ qh).astype(np.float32) * scale
                scores -= np.max(scores)
                att = np.exp(scores).astype(np.float32)
                att /= np.sum(att, dtype=np.float32)
                vh = self.value_cache[layer, : pos + 1, (h // kv_mul) * hs : (h // kv_mul + 1) * hs]
                heads[h] = att @ vh
            x = (x + self.wo[layer] @ heads.reshape(c.dim)).astype(np.float32)
            xb = _rmsnorm(x, self.rms_ffn[layer])
            h1 = (self.w1[layer] @ xb).astype(np.float32)
            h3 = (self.w3[layer] @ xb).astype(np.float32)
            h1 = (h1 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-h1).astype(np.float32))) * h3).astype(np.float32)
            x = (x + self.w2[layer] @ h1).astype(np.float32)
        x = _rmsnorm(x, self.rms_final)
        return (self.wcls @ x).astype(np.float32)
