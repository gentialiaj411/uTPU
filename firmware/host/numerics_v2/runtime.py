"""NumPy integer-only single-token llama2.c decode.

No floating-point literal, dtype, conversion, or arithmetic is present here.
All ratios arrive as integer multiplier/right-shift records from export.py.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

from .fixed import checked_matmul, div_rne, requant, rshift_rne


def _linear(x, record):
    weights = record["weight"]
    groups = record["groups"]
    values = np.asarray(x)
    out = np.zeros(values.shape[:-1] + (weights.shape[0],), dtype=np.int64)
    qmax = int(record["qmax"])
    for group in range(groups.shape[1]):
        begin = int(groups[0, group])
        end = int(groups[1, group])
        accum = checked_matmul(values[..., begin:end], weights[:, begin:end], end - begin, qmax)
        out += rshift_rne(accum.astype(np.int64) * record["multiplier"][:, group], record["shift"][:, group])
    return np.clip(out, -128, 127).astype(np.int8)


def _rmsnorm(x, gamma, gamma_fraction_bits, out_multiplier, out_shift):
    square_sum = int(np.sum(np.asarray(x, dtype=np.int32) ** 2, dtype=np.int32))
    mean = int(div_rne(square_sum, x.size)) + 1
    root_q15 = max(1, math.isqrt(mean << 30))
    normalized_q15 = div_rne(np.asarray(x, dtype=np.int64) << 30, root_q15)
    combined = rshift_rne(normalized_q15 * np.asarray(gamma, dtype=np.int64), gamma_fraction_bits)
    return requant(combined, out_multiplier, out_shift)


def _rmsnorm_rows(x, gamma, gamma_fraction_bits, out_multiplier, out_shift):
    values = np.asarray(x, dtype=np.int32)
    square_sum = np.sum(values ** 2, axis=-1, dtype=np.int32)
    mean = div_rne(square_sum, values.shape[-1]).astype(np.int64) + 1
    roots_q15 = np.fromiter((max(1, math.isqrt(int(item) << 30)) for item in mean.reshape(-1)), dtype=np.int64).reshape(mean.shape)
    normalized_q15 = div_rne(values.astype(np.int64) << 30, roots_q15[..., None])
    combined = rshift_rne(normalized_q15 * np.asarray(gamma, dtype=np.int64), gamma_fraction_bits)
    return requant(combined, out_multiplier, out_shift)


class IntegerLlama:
    def __init__(self, export, lut_dir):
        self.cfg = export["config"]
        self.embedding = export["embedding"]
        self.gamma_att = export["gamma_att"]
        self.gamma_ffn = export["gamma_ffn"]
        self.gamma_final = export["gamma_final"]
        self.gamma_fraction_bits = int(export["gamma_fraction_bits"])
        self.layers = export["layers"]
        self.classifier = export["classifier"]
        self.norm_multiplier = int(export["norm_multiplier"])
        self.norm_shift = int(export["norm_shift"])
        self.score_multiplier = int(export["score_multiplier"])
        self.score_shift = int(export["score_shift"])
        self.swiglu_multiplier = int(export["swiglu_multiplier"])
        self.swiglu_shift = int(export["swiglu_shift"])
        self.residual_multiplier = np.asarray(export["residual_multiplier"], dtype=np.int64)
        self.residual_shift = np.asarray(export["residual_shift"], dtype=np.int64)
        folder = Path(lut_dir)
        luts = json.loads((folder / "luts.json").read_text(encoding="utf-8"))
        self.rope_cos = np.asarray(luts["rope_cos_q15"]["values"], dtype=np.int16)
        self.rope_sin = np.asarray(luts["rope_sin_q15"]["values"], dtype=np.int16)
        self.silu = np.asarray(luts["silu_i8"]["values"], dtype=np.int8)
        self.exp = np.asarray(luts["exp_q15"]["values"], dtype=np.uint16)
        layers, seq, kv = int(self.cfg["n_layers"]), int(self.cfg["seq_len"]), int(self.cfg["kv_dim"])
        self.key_cache = np.zeros((layers, seq, kv), dtype=np.int8)
        self.value_cache = np.zeros_like(self.key_cache)

    def reset(self):
        self.key_cache.fill(0)
        self.value_cache.fill(0)

    def _rope(self, vector, position):
        pair = vector.reshape(-1, 2).astype(np.int32)
        cos = self.rope_cos[position, np.arange(pair.shape[0]) % self.rope_cos.shape[1]].astype(np.int32)
        sin = self.rope_sin[position, np.arange(pair.shape[0]) % self.rope_sin.shape[1]].astype(np.int32)
        first = rshift_rne(pair[:, 0] * cos - pair[:, 1] * sin, 15)
        second = rshift_rne(pair[:, 0] * sin + pair[:, 1] * cos, 15)
        return np.clip(np.stack((first, second), axis=1), -128, 127).astype(np.int8).reshape(-1)

    def _softmax(self, score):
        delta = np.clip(score.astype(np.int16) - int(np.max(score)), -128, 0)
        exponent = self.exp[(-delta).astype(np.int16)].astype(np.int32)
        total = int(np.sum(exponent, dtype=np.int32))
        return div_rne(exponent.astype(np.int64) * 32767, total).astype(np.int16)

    def _rope_sequence(self, vector):
        pair = vector.reshape(vector.shape[0], -1, 2).astype(np.int32)
        cos = self.rope_cos[:vector.shape[0], np.arange(pair.shape[1]) % self.rope_cos.shape[1]].astype(np.int32)
        sin = self.rope_sin[:vector.shape[0], np.arange(pair.shape[1]) % self.rope_sin.shape[1]].astype(np.int32)
        first = rshift_rne(pair[:, :, 0] * cos - pair[:, :, 1] * sin, 15)
        second = rshift_rne(pair[:, :, 0] * sin + pair[:, :, 1] * cos, 15)
        return np.clip(np.stack((first, second), axis=2), -128, 127).astype(np.int8).reshape(vector.shape)

    def forward_sequence(self, tokens):
        """Vectorized causal teacher-forced path; integer-equivalent to cached decode."""
        dim, heads = int(self.cfg["dim"]), int(self.cfg["n_heads"])
        kv_heads, head_size = int(self.cfg["n_kv_heads"]), int(self.cfg["head_size"])
        kv_mul = heads // kv_heads
        ids = np.asarray(tokens, dtype=np.int64)
        if ids.ndim != 1 or ids.size > int(self.cfg["seq_len"]):
            raise ValueError("tokens must be a 1-D sequence within configured length")
        x = self.embedding[ids].copy()
        causal = np.triu(np.ones((ids.size, ids.size), dtype=np.bool_), 1)
        for layer_id, layer in enumerate(self.layers):
            norm = _rmsnorm_rows(x, self.gamma_att[layer_id], self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
            q = self._rope_sequence(_linear(norm, layer["wq"]))
            k = self._rope_sequence(_linear(norm, layer["wk"]))
            v = _linear(norm, layer["wv"])
            heads_out = np.empty((ids.size, dim), dtype=np.int8)
            for head in range(heads):
                begin = (head // kv_mul) * head_size
                end = begin + head_size
                score_acc = checked_matmul(q[:, head * head_size:(head + 1) * head_size], k[:, begin:end], head_size, 127)
                score = requant(score_acc, self.score_multiplier, self.score_shift)
                masked = np.where(causal, -128, score).astype(np.int16)
                delta = np.clip(masked - np.max(masked, axis=1, keepdims=True), -128, 0)
                exponent = self.exp[(-delta).astype(np.int16)].astype(np.int32)
                exponent[causal] = 0
                total = np.sum(exponent, axis=1, dtype=np.int32)
                probability = div_rne(exponent.astype(np.int64) * 32767, total[:, None]).astype(np.int32)
                assert ids.size * 32767 * 127 <= (1 << 31) - 1
                av = probability @ v[:, begin:end].astype(np.int32)
                assert av.dtype == np.int32
                heads_out[:, head * head_size:(head + 1) * head_size] = np.clip(rshift_rne(av, 15), -128, 127).astype(np.int8)
            branch = _linear(heads_out, layer["wo"])
            x = requant(x.astype(np.int32) + branch.astype(np.int32), self.residual_multiplier[0], self.residual_shift[0])
            norm = _rmsnorm_rows(x, self.gamma_ffn[layer_id], self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
            h1 = _linear(norm, layer["w1"])
            h3 = _linear(norm, layer["w3"])
            silu = self.silu[h1.astype(np.int16) + 128]
            hidden = requant(silu.astype(np.int32) * h3.astype(np.int32), self.swiglu_multiplier, self.swiglu_shift)
            branch = _linear(hidden, layer["w2"])
            x = requant(x.astype(np.int32) + branch.astype(np.int32), self.residual_multiplier[1], self.residual_shift[1])
        norm = _rmsnorm_rows(x, self.gamma_final, self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
        return _linear(norm, self.classifier)

    def forward(self, token, position):
        dim, heads = int(self.cfg["dim"]), int(self.cfg["n_heads"])
        kv_heads, head_size = int(self.cfg["n_kv_heads"]), int(self.cfg["head_size"])
        kv_mul = heads // kv_heads
        x = self.embedding[int(token)].copy()
        for layer_id, layer in enumerate(self.layers):
            norm = _rmsnorm(x, self.gamma_att[layer_id], self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
            q = self._rope(_linear(norm, layer["wq"]), position)
            k = self._rope(_linear(norm, layer["wk"]), position)
            v = _linear(norm, layer["wv"])
            self.key_cache[layer_id, position] = k
            self.value_cache[layer_id, position] = v
            heads_out = np.empty(dim, dtype=np.int8)
            for head in range(heads):
                begin = (head // kv_mul) * head_size
                end = begin + head_size
                keys = self.key_cache[layer_id, : position + 1, begin:end]
                score_acc = checked_matmul(q[head * head_size:(head + 1) * head_size], keys, head_size, 127)
                score = requant(score_acc, self.score_multiplier, self.score_shift)
                probability = self._softmax(score)
                values = self.value_cache[layer_id, : position + 1, begin:end].astype(np.int32)
                av = probability.astype(np.int32) @ values
                heads_out[head * head_size:(head + 1) * head_size] = np.clip(rshift_rne(av, 15), -128, 127).astype(np.int8)
            branch = _linear(heads_out, layer["wo"])
            x = requant(x.astype(np.int32) + branch.astype(np.int32), self.residual_multiplier[0], self.residual_shift[0])
            norm = _rmsnorm(x, self.gamma_ffn[layer_id], self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
            h1 = _linear(norm, layer["w1"])
            h3 = _linear(norm, layer["w3"])
            silu = self.silu[h1.astype(np.int16) + 128]
            hidden = requant(silu.astype(np.int32) * h3.astype(np.int32), self.swiglu_multiplier, self.swiglu_shift)
            branch = _linear(hidden, layer["w2"])
            x = requant(x.astype(np.int32) + branch.astype(np.int32), self.residual_multiplier[1], self.residual_shift[1])
        norm = _rmsnorm(x, self.gamma_final, self.gamma_fraction_bits, self.norm_multiplier, self.norm_shift)
        return _linear(norm, self.classifier)
