"""Pure-integer runq-style path: dynamic INT8 matmul operands, INT32 boundaries."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fixed import checked_matmul, div127, div_const_rne, mul_recip, reciprocal_rne, rshift_rne

ACTIVATION_GROUP = 64
ATTENTION_VALUE_GROUP = 16


@dataclass
class QValue:
    code: np.ndarray
    scale_multiplier: np.ndarray
    group_size: int = ACTIVATION_GROUP
    qmax: int = 127


def _quantize(raw, group_size=ACTIVATION_GROUP, qmax=127):
    value = np.asarray(raw, dtype=np.int64)
    codes, scales = [], []
    for begin in range(0, value.shape[-1], group_size):
        block = value[..., begin:begin + group_size]
        peak = np.maximum(1, np.max(np.abs(block), axis=-1)).astype(np.int32)
        reciprocal = reciprocal_rne(qmax, peak)
        codes.append(np.clip(mul_recip(block, reciprocal[..., None]), -qmax, qmax).astype(np.int8))
        scales.append(peak)
    return QValue(np.concatenate(codes, axis=-1), np.stack(scales, axis=-1), group_size, qmax)


def _to_base(value):
    chunks = []
    for group, begin in enumerate(range(0, value.code.shape[-1], value.group_size)):
        block = value.code[..., begin:begin + value.group_size].astype(np.int64)
        chunks.append(div_const_rne(block * value.scale_multiplier[..., group, None], value.qmax))
    return np.concatenate(chunks, axis=-1)


def _small_multiplier(multiplier):
    return rshift_rne(np.asarray(multiplier, dtype=np.int64), 16)


def _linear_high(value, record, output_begin=0, output_end=None):
    output_end = record["weight"].shape[0] if output_end is None else output_end
    weights, groups = record["weight"][output_begin:output_end], record["groups"]
    multipliers = record["multiplier"][output_begin:output_end]
    shifts = record["shift"][output_begin:output_end]
    raw = np.zeros(value.code.shape[:-1] + (weights.shape[0],), dtype=np.int64)
    for weight_group in range(groups.shape[1]):
        group_begin, group_end = int(groups[0, weight_group]), int(groups[1, weight_group])
        for begin in range(group_begin, group_end, ACTIVATION_GROUP):
            end = min(begin + ACTIVATION_GROUP, group_end)
            accum = checked_matmul(value.code[..., begin:end], weights[:, begin:end], end - begin, int(record["qmax"]))
            factor = value.scale_multiplier[..., begin // ACTIVATION_GROUP, None]
            product = accum.astype(np.int64) * factor * _small_multiplier(multipliers[:, weight_group])
            scaled = rshift_rne(product, shifts[:, weight_group] - 16)
            raw += div127(scaled)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise OverflowError("linear high-precision output exceeds INT32")
    return raw.astype(np.int32)


def _rms_high(value, gamma, gamma_fraction_bits, multiplier, shift, epsilon):
    x = np.asarray(value, dtype=np.int64)
    square_sum = np.sum(x * x, axis=-1, dtype=np.int64)
    # Hidden widths are fixed model constants; use reciprocal-multiply rather
    # than a per-element divide.  The exactness proof is exercised in tests.
    mean = div_const_rne(square_sum, x.shape[-1]).astype(np.int64) + int(epsilon)
    roots = np.fromiter((max(1, math.isqrt(int(item) << 30)) for item in mean.reshape(-1)), dtype=np.int64).reshape(mean.shape)
    reciprocal = reciprocal_rne(1 << 30, roots)
    normalized = mul_recip(x, reciprocal[..., None])
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
            reciprocal = reciprocal_rne(127, peak)
            code = np.clip(mul_recip(block, reciprocal), -128, 127).astype(np.int32)
            accum = prob[begin:begin + width] @ code
        else:
            # Each teacher-forced row must use only its causal prefix when
            # choosing a block scale. Future masked values cannot affect it.
            token_peak = np.max(np.abs(block), axis=-1)
            prefix_peak = np.maximum(1, np.maximum.accumulate(token_peak))
            peak = np.ones(prob.shape[-2], dtype=np.int64)
            peak[begin:begin + width] = prefix_peak
            peak[begin + width:] = prefix_peak[-1]
            reciprocal = reciprocal_rne(127, peak)
            code = np.clip(mul_recip(block[None, ...], reciprocal[:, None, None]),
                           -128, 127).astype(np.int32)
            lhs = prob[..., begin:begin + width][..., None, :]
            accum = (lhs @ code).squeeze(-2)
        assert accum.dtype == np.int32
        weighted = rshift_rne(accum.astype(np.int64) * np.asarray(peak)[..., None], 15)
        raw += div127(weighted)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise OverflowError("attention value exceeds INT32")
    return raw.astype(np.int32)


class RunQIntegerLlama:
    logits_are_base_units = True

    def __init__(self, export, lut_dir, kv_bits=8):
        self.cfg, self.embedding = export["config"], export["embedding"]
        if kv_bits not in (4, 8):
            raise ValueError("kv_bits must be 4 or 8")
        self.kv_bits = int(kv_bits)
        self.embedding_tied = bool(export.get("embedding_tied", False))
        if self.embedding_tied:
            self.embedding_group_multiplier = export["embedding_group_multiplier"]
            self.embedding_group_shift = export["embedding_group_shift"]
            self.embedding_group_size = int(export["embedding_group_size"])
        else:
            self.embedding_scale_multiplier = export["embedding_scale_multiplier"]
            self.embedding_scale_shift = int(export["embedding_scale_shift"])
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
        head_size = int(self.cfg["head_size"])
        self.rope_cos = np.asarray(luts[f"rope_cos_q15_h{head_size}"]["values"], dtype=np.int16)
        self.rope_sin = np.asarray(luts[f"rope_sin_q15_h{head_size}"]["values"], dtype=np.int16)
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
        if self.embedding_tied:
            chunks = []
            for group, begin in enumerate(range(0, code.shape[-1], self.embedding_group_size)):
                end = min(begin + self.embedding_group_size, code.shape[-1])
                chunks.append(rshift_rne(code[..., begin:end] *
                                         self.embedding_group_multiplier[ids, group, None],
                                         self.embedding_group_shift[ids, group, None]))
            return np.concatenate(chunks, axis=-1).astype(np.int32)
        peak = self.embedding_scale_multiplier[ids]
        return rshift_rne(code * np.asarray(peak)[..., None], self.embedding_scale_shift).astype(np.int32)

    def _rope_high(self, value, positions):
        x = np.asarray(value, dtype=np.int64)
        pair = x.reshape(x.shape[:-1] + (-1, 2))
        index = np.arange(pair.shape[-2]) % (int(self.cfg["head_size"]) // 2)
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
        reciprocal = reciprocal_rne(32767, total)
        return mul_recip(exponent, reciprocal[..., None]).astype(np.int32)

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

    def _score(self, q, k8):
        q8 = _quantize(q)
        accum = checked_matmul(q8.code, k8.code, q8.code.shape[-1], 127)
        product = accum.astype(np.int64) * q8.scale_multiplier[..., 0, None] * k8.scale_multiplier[..., 0]
        product *= _small_multiplier(self.score_multiplier)
        score = rshift_rne(product, self.score_shift - 16)
        # q is always INT8 (qmax 127); k uses the KV-cache qmax (127 for INT8, 7 for INT4).
        score = div127(score)
        score = div127(score) if k8.qmax == 127 else div_const_rne(score, k8.qmax)
        if np.any(score < -(1 << 31)) or np.any(score > (1 << 31) - 1):
            raise OverflowError("attention score exceeds INT32")
        return score.astype(np.int32)

    def _forward_sequence_hidden(self, tokens):
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
            qmax = 127 if self.kv_bits == 8 else 7
            k8, v8 = _quantize(k, hs, qmax), _quantize(v, hs, qmax)
            v_base = _to_base(v8).astype(np.int32)
            heads_high = np.empty((length, int(c["dim"])), dtype=np.int32)
            for head in range(heads):
                kv_head = head // kv_mul
                begin, end = kv_head * hs, (kv_head + 1) * hs
                key = QValue(k8.code[:, begin:end], k8.scale_multiplier[:, kv_head, None], hs, k8.qmax)
                score = self._score(q[:, head * hs:(head + 1) * hs], key)
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
        return _quantize(norm)

    def forward_sequence(self, tokens):
        return _linear_high(self._forward_sequence_hidden(tokens), self.classifier)

    def forward_sequence_logits_chunks(self, tokens, chunk_size):
        hidden = self._forward_sequence_hidden(tokens)
        vocab = int(self.cfg["vocab_size"])
        for begin in range(0, vocab, chunk_size):
            end = min(begin + chunk_size, vocab)
            yield begin, _linear_high(hidden, self.classifier, begin, end)

    def _forward_hidden(self, token, position):
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
            qmax = 127 if self.kv_bits == 8 else 7
            k8, v8 = _quantize(k, hs, qmax), _quantize(v, hs, qmax)
            self.key_code[layer_id, position] = k8.code
            self.value_code[layer_id, position] = v8.code
            self.key_scale[layer_id, position] = k8.scale_multiplier
            self.value_scale[layer_id, position] = v8.scale_multiplier
            heads_high = np.empty(int(c["dim"]), dtype=np.int32)
            for head in range(heads):
                kv_head = head // kv_mul
                begin, end = kv_head * hs, (kv_head + 1) * hs
                key = QValue(self.key_code[layer_id, :position + 1, begin:end],
                             self.key_scale[layer_id, :position + 1, kv_head, None], hs,
                             127 if self.kv_bits == 8 else 7)
                value = QValue(self.value_code[layer_id, :position + 1, begin:end],
                               self.value_scale[layer_id, :position + 1, kv_head, None], hs,
                               127 if self.kv_bits == 8 else 7)
                score = self._score(q[None, head * hs:(head + 1) * hs], key)[0]
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
        return _quantize(norm)

    def forward(self, token, position):
        return _linear_high(self._forward_hidden(token, position), self.classifier)

    def forward_topk(self, token, position, k=1, chunk_size=1024):
        if not (1 <= k <= int(self.cfg["vocab_size"])):
            raise ValueError("invalid top-k")
        hidden = self._forward_hidden(token, position)
        best_values = np.empty(0, dtype=np.int32)
        best_indices = np.empty(0, dtype=np.int64)
        vocab = int(self.cfg["vocab_size"])
        for begin in range(0, vocab, chunk_size):
            end = min(begin + chunk_size, vocab)
            values = _linear_high(hidden, self.classifier, begin, end)
            indices = np.arange(begin, end, dtype=np.int64)
            merged_values = np.concatenate((best_values, values))
            merged_indices = np.concatenate((best_indices, indices))
            order = np.lexsort((merged_indices, -merged_values.astype(np.int64)))[:k]
            best_values, best_indices = merged_values[order], merged_indices[order]
        return best_indices, best_values

    def forward_argmax(self, token, position, chunk_size=1024):
        indices, _ = self.forward_topk(token, position, 1, chunk_size)
        return int(indices[0])
