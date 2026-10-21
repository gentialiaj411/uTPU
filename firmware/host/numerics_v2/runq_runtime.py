"""Pure-integer runq-style path: dynamic INT8 matmul operands, INT32 boundaries."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fixed import checked_matmul, div_rne, rshift_rne

ACTIVATION_GROUP = 64
ATTENTION_VALUE_GROUP = 16


@dataclass
class QValue:
    code: np.ndarray
    scale_multiplier: np.ndarray
    group_size: int = ACTIVATION_GROUP


def _quantize(raw, group_size=ACTIVATION_GROUP):
    value = np.asarray(raw, dtype=np.int64)
    codes, scales = [], []
    for begin in range(0, value.shape[-1], group_size):
        block = value[..., begin:begin + group_size]
        peak = np.maximum(1, np.max(np.abs(block), axis=-1)).astype(np.int32)
        codes.append(np.clip(div_rne(block << 7, peak[..., None]), -128, 127).astype(np.int8))
        scales.append(peak)
    return QValue(np.concatenate(codes, axis=-1), np.stack(scales, axis=-1), group_size)


def _to_base(value):
    chunks = []
    for group, begin in enumerate(range(0, value.code.shape[-1], value.group_size)):
        block = value.code[..., begin:begin + value.group_size].astype(np.int64)
        chunks.append(rshift_rne(block * value.scale_multiplier[..., group, None], 7))
    return np.concatenate(chunks, axis=-1)


def _small_multiplier(multiplier):
    return rshift_rne(np.asarray(multiplier, dtype=np.int64), 16)


def _linear_high(value, record):
    weights, groups = record["weight"], record["groups"]
    raw = np.zeros(value.code.shape[:-1] + (weights.shape[0],), dtype=np.int64)
    for weight_group in range(groups.shape[1]):
        group_begin, group_end = int(groups[0, weight_group]), int(groups[1, weight_group])
        for begin in range(group_begin, group_end, ACTIVATION_GROUP):
            end = min(begin + ACTIVATION_GROUP, group_end)
            accum = checked_matmul(value.code[..., begin:end], weights[:, begin:end], end - begin, int(record["qmax"]))
            factor = value.scale_multiplier[..., begin // ACTIVATION_GROUP, None]
            product = accum.astype(np.int64) * factor * _small_multiplier(record["multiplier"][:, weight_group])
            raw += rshift_rne(product, record["shift"][:, weight_group] - 16 + 7)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise OverflowError("linear high-precision output exceeds INT32")
    return raw.astype(np.int32)


def _rms_high(value, gamma, gamma_fraction_bits, multiplier, shift, epsilon):
    x = np.asarray(value, dtype=np.int64)
    square_sum = np.sum(x * x, axis=-1, dtype=np.int64)
    mean = div_rne(square_sum, x.shape[-1]).astype(np.int64) + int(epsilon)
    roots = np.fromiter((max(1, math.isqrt(int(item) << 30)) for item in mean.reshape(-1)), dtype=np.int64).reshape(mean.shape)
    normalized = div_rne(x << 30, roots[..., None])
    combined = rshift_rne(normalized * np.asarray(gamma, dtype=np.int64), gamma_fraction_bits)
    out = rshift_rne(combined * int(multiplier), int(shift))
    return out.astype(np.int32)


def _attention_value(probability, value):
    """Q15-by-dynamic-INT8 block matmul with proven INT32 dot products."""
    prob = np.asarray(probability, dtype=np.int32)
    values = np.asarray(value, dtype=np.int64)
    raw = np.zeros(prob.shape[:-1] + (values.shape[-1],), dtype=np.int64)
    for begin in range(0, values.shape[-2], ATTENTION_VALUE_GROUP):
        block = values[begin:begin + ATTENTION_VALUE_GROUP]
        width = block.shape[0]
        bound = width * 32767 * 127
        assert bound <= (1 << 31) - 1, (width, bound)
        if prob.ndim == 1:
            peak = max(1, int(np.max(np.abs(block))))
            code = np.clip(div_rne(block << 7, peak), -128, 127).astype(np.int32)
            accum = prob[begin:begin + width] @ code
        else:
            # Each teacher-forced row must use only its causal prefix when
            # choosing a block scale. Future masked values cannot affect it.
            token_peak = np.max(np.abs(block), axis=-1)
            prefix_peak = np.maximum(1, np.maximum.accumulate(token_peak))
            peak = np.ones(prob.shape[-2], dtype=np.int64)
            peak[begin:begin + width] = prefix_peak
            peak[begin + width:] = prefix_peak[-1]
            code = np.clip(div_rne(block[None, ...] << 7, peak[:, None, None]),
                           -128, 127).astype(np.int32)
            lhs = prob[..., begin:begin + width][..., None, :]
            accum = (lhs @ code).squeeze(-2)
        assert accum.dtype == np.int32
        raw += rshift_rne(accum.astype(np.int64) * np.asarray(peak)[..., None], 22)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise OverflowError("attention value exceeds INT32")
    return raw.astype(np.int32)


class RunQIntegerLlama:
    logits_are_base_units = True

    def __init__(self, export, lut_dir):
        self.cfg, self.embedding = export["config"], export["embedding"]
        self.embedding_scale_multiplier = export["embedding_scale_multiplier"]
        self.gamma_att, self.gamma_ffn, self.gamma_final = export["gamma_att"], export["gamma_ffn"], export["gamma_final"]
        self.layers, self.classifier = export["layers"], export["classifier"]
        self.norm_multiplier, self.norm_shift = int(export["norm_multiplier"]), int(export["norm_shift"])
        self.gamma_fraction_bits = int(export["gamma_fraction_bits"])
        self.score_multiplier, self.score_shift = int(export["score_multiplier"]), int(export["score_shift"])
        self.swiglu_multiplier, self.swiglu_shift = int(export["swiglu_multiplier"]), int(export["swiglu_shift"])
        self.silu_index_multiplier = int(export["silu_index_multiplier"])
        self.silu_index_shift = int(export["silu_index_shift"])
        self.rms_epsilon_x = int(export["rms_epsilon_x"])
        luts = json.loads((Path(lut_dir) / "luts.json").read_text(encoding="utf-8"))
        self.rope_cos = np.asarray(luts["rope_cos_q15"]["values"], dtype=np.int16)
        self.rope_sin = np.asarray(luts["rope_sin_q15"]["values"], dtype=np.int16)
        self.exp = np.asarray(luts["exp_q15"]["values"], dtype=np.uint16)
        self.silu = np.asarray(luts["silu_i8"]["values"], dtype=np.int8)
        self.sigmoid = np.asarray(luts["sigmoid_q15"]["values"], dtype=np.uint16)
        layers, seq, kv = int(self.cfg["n_layers"]), int(self.cfg["seq_len"]), int(self.cfg["kv_dim"])
        self.key_code = np.zeros((layers, seq, kv), dtype=np.int8)
        self.value_code = np.zeros_like(self.key_code)
        self.key_scale = np.ones((layers, seq, int(self.cfg["n_kv_heads"])), dtype=np.int32)
        self.value_scale = np.ones_like(self.key_scale)
        self.tokens = []

    def reset(self):
        self.tokens.clear()

    def _embedding_high(self, ids):
        code = self.embedding[ids].astype(np.int64)
        peak = self.embedding_scale_multiplier[ids]
        return rshift_rne(code * np.asarray(peak)[..., None], 7).astype(np.int32)

    def _rope_high(self, value, positions):
        x = np.asarray(value, dtype=np.int64)
        pair = x.reshape(x.shape[:-1] + (-1, 2))
        index = np.arange(pair.shape[-2]) % self.rope_cos.shape[1]
        cos, sin = self.rope_cos[positions, index].astype(np.int64), self.rope_sin[positions, index].astype(np.int64)
        out = np.stack((rshift_rne(pair[..., 0] * cos - pair[..., 1] * sin, 15),
                        rshift_rne(pair[..., 0] * sin + pair[..., 1] * cos, 15)), axis=-1)
        return out.reshape(x.shape).astype(np.int32)

    def _softmax(self, score, causal=None):
        # Preserve the wide absolute score until after stable-softmax max
        # subtraction.  Only the non-positive delta is narrowed to the LUT
        # domain; narrowing the absolute score changes ordering on saturation.
        wide = score.astype(np.int64)
        masked = wide if causal is None else np.where(causal, -(1 << 60), wide)
        delta = np.clip(masked - np.max(masked, axis=-1, keepdims=True), -1024, 0)
        exponent = self.exp[(-delta).astype(np.int16)].astype(np.int32)
        if causal is not None:
            exponent[causal] = 0
        total = np.sum(exponent, axis=-1, dtype=np.int32)
        return div_rne(exponent.astype(np.int64) * 32767, total[..., None]).astype(np.int32)

    def _silu_high(self, value):
        index = rshift_rne(value.astype(np.int64) * self.silu_index_multiplier,
                           self.silu_index_shift)
        sigmoid = np.where(index < -512, 0,
                           np.where(index > 512, 32767,
                                    self.sigmoid[np.clip(index, -512, 512) + 512])).astype(np.int64)
        return rshift_rne(value.astype(np.int64) * sigmoid, 15).astype(np.int32)

    def _hidden_high(self, silu, h3):
        product = silu.astype(np.int64) * h3.astype(np.int64) * _small_multiplier(self.swiglu_multiplier)
        return rshift_rne(product, self.swiglu_shift - 16).astype(np.int32)

    def _score(self, q, k):
        q8, k8 = _quantize(q), _quantize(k)
        accum = checked_matmul(q8.code, k8.code, q8.code.shape[-1], 127)
        product = accum.astype(np.int64) * q8.scale_multiplier[..., 0, None] * k8.scale_multiplier[..., 0]
        product *= _small_multiplier(self.score_multiplier)
        score = rshift_rne(product, self.score_shift - 16 + 14)
        if np.any(score < -(1 << 31)) or np.any(score > (1 << 31) - 1):
            raise OverflowError("attention score exceeds INT32")
        return score.astype(np.int32)

    def forward_sequence(self, tokens):
        ids = np.asarray(tokens, dtype=np.int64)
        c, length = self.cfg, ids.size
        if ids.ndim != 1 or length > int(c["seq_len"]):
            raise ValueError("invalid token sequence")
        x = self._embedding_high(ids)
        heads, kv_heads, hs = int(c["n_heads"]), int(c["n_kv_heads"]), int(c["head_size"])
        kv_mul = heads // kv_heads
        causal = np.triu(np.ones((length, length), dtype=np.bool_), 1)
        positions = np.arange(length)[:, None]
        for layer_id, layer in enumerate(self.layers):
            norm = _rms_high(x, self.gamma_att[layer_id], self.gamma_fraction_bits,
                             self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
            norm8 = _quantize(norm)
            q = self._rope_high(_linear_high(norm8, layer["wq"]), positions)
            k = self._rope_high(_linear_high(norm8, layer["wk"]), positions)
            v = _linear_high(norm8, layer["wv"])
            # KV cache gets an independent scale per KV head.  Sharing one
            # scale across all heads lets one outlier discard useful bits in
            # unrelated heads.
            k8, v8 = _quantize(k, hs), _quantize(v, hs)
            k_base, v_base = _to_base(k8).astype(np.int32), _to_base(v8).astype(np.int32)
            heads_high = np.empty((length, int(c["dim"])), dtype=np.int32)
            for head in range(heads):
                begin, end = (head // kv_mul) * hs, (head // kv_mul + 1) * hs
                score = self._score(q[:, head * hs:(head + 1) * hs], k_base[:, begin:end])
                probability = self._softmax(score, causal)
                heads_high[:, head * hs:(head + 1) * hs] = _attention_value(
                    probability, v_base[:, begin:end])
            x = (x.astype(np.int64) + _linear_high(_quantize(heads_high), layer["wo"])).astype(np.int32)
            norm = _rms_high(x, self.gamma_ffn[layer_id], self.gamma_fraction_bits,
                             self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
            norm8 = _quantize(norm)
            h1, h3 = _linear_high(norm8, layer["w1"]), _linear_high(norm8, layer["w3"])
            hidden = self._hidden_high(self._silu_high(h1), h3)
            x = (x.astype(np.int64) + _linear_high(_quantize(hidden), layer["w2"])).astype(np.int32)
        norm = _rms_high(x, self.gamma_final, self.gamma_fraction_bits,
                         self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
        return _linear_high(_quantize(norm), self.classifier)

    def forward(self, token, position):
        if position != len(self.tokens):
            raise ValueError("positions must be contiguous")
        self.tokens.append(int(token))
        c = self.cfg
        x = self._embedding_high(np.asarray(token, dtype=np.int64))
        heads, kv_heads, hs = int(c["n_heads"]), int(c["n_kv_heads"]), int(c["head_size"])
        kv_mul = heads // kv_heads
        for layer_id, layer in enumerate(self.layers):
            norm = _rms_high(x, self.gamma_att[layer_id], self.gamma_fraction_bits,
                             self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
            norm8 = _quantize(norm)
            q = self._rope_high(_linear_high(norm8, layer["wq"]), position)
            k = self._rope_high(_linear_high(norm8, layer["wk"]), position)
            v = _linear_high(norm8, layer["wv"])
            k8, v8 = _quantize(k, hs), _quantize(v, hs)
            self.key_code[layer_id, position] = k8.code
            self.value_code[layer_id, position] = v8.code
            self.key_scale[layer_id, position] = k8.scale_multiplier
            self.value_scale[layer_id, position] = v8.scale_multiplier
            heads_high = np.empty(int(c["dim"]), dtype=np.int32)
            for head in range(heads):
                kv_head = head // kv_mul
                begin, end = kv_head * hs, (kv_head + 1) * hs
                key = QValue(self.key_code[layer_id, :position + 1, begin:end],
                             self.key_scale[layer_id, :position + 1, kv_head, None], hs)
                value = QValue(self.value_code[layer_id, :position + 1, begin:end],
                               self.value_scale[layer_id, :position + 1, kv_head, None], hs)
                score = self._score(q[None, head * hs:(head + 1) * hs], _to_base(key))[0]
                probability = self._softmax(score)
                heads_high[head * hs:(head + 1) * hs] = _attention_value(
                    probability, _to_base(value))
            x = (x.astype(np.int64) + _linear_high(_quantize(heads_high), layer["wo"])).astype(np.int32)
            norm = _rms_high(x, self.gamma_ffn[layer_id], self.gamma_fraction_bits,
                             self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
            norm8 = _quantize(norm)
            h1, h3 = _linear_high(norm8, layer["w1"]), _linear_high(norm8, layer["w3"])
            hidden = self._hidden_high(self._silu_high(h1), h3)
            x = (x.astype(np.int64) + _linear_high(_quantize(hidden), layer["w2"])).astype(np.int32)
        norm = _rms_high(x, self.gamma_final, self.gamma_fraction_bits,
                         self.norm_multiplier, self.norm_shift, self.rms_epsilon_x)
        return _linear_high(_quantize(norm), self.classifier)
