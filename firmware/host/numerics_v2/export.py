"""Offline float-to-integer export for numerics-v2."""

from __future__ import annotations

import math

import numpy as np


def multiplier_shift(ratio: float):
    if ratio < 0.0:
        raise ValueError("scale ratio must be nonnegative")
    if ratio == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(ratio)
    multiplier = int(round(mantissa * (1 << 30)))
    if multiplier == 1 << 30:
        multiplier >>= 1
        exponent += 1
    shift = 30 - exponent
    if shift < 0:
        multiplier <<= -shift
        shift = 0
    if multiplier > (1 << 31) - 1:
        raise OverflowError((ratio, multiplier, shift))
    return multiplier, shift


def _clip_for_mse(values, qmax):
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return 1.0
    best = (float("inf"), peak)
    for fraction in np.linspace(0.55, 1.0, 46):
        clip = peak * float(fraction)
        scale = clip / qmax
        code = np.clip(np.rint(values / scale), -qmax, qmax)
        error = float(np.mean((values - code * scale) ** 2))
        if error < best[0]:
            best = (error, clip)
    return best[1]


def quantized_linear(weight, input_scale, output_scale, bits, group_size, improved=False):
    value = np.asarray(weight, dtype=np.float32)
    out, width = value.shape
    qmax = 127 if bits == 8 else 7
    size = group_size
    ranges = [(start, min(start + size, width)) for start in range(0, width, size)]
    codes = np.empty_like(value, dtype=np.int8)
    multipliers = np.empty((out, len(ranges)), dtype=np.int64)
    shifts = np.empty((out, len(ranges)), dtype=np.int64)
    for row in range(out):
        for group, (begin, end) in enumerate(ranges):
            block = value[row, begin:end]
            clip = _clip_for_mse(block, qmax) if improved and bits == 4 else float(np.max(np.abs(block)))
            scale = clip / qmax if clip else 1.0
            codes[row, begin:end] = np.clip(np.rint(block / scale), -qmax, qmax).astype(np.int8)
            multipliers[row, group], shifts[row, group] = multiplier_shift(input_scale * scale / output_scale)
    return {"weight": codes, "multiplier": multipliers, "shift": shifts,
            "groups": np.asarray([[a for a, _ in ranges], [b for _, b in ranges]], dtype=np.int32),
            "qmax": qmax}


def export_model(reference, scales, bits=8, group_size=32, improved=False):
    c = reference.config
    embedding_scale = scales["x"]
    embedding = np.clip(np.rint(np.asarray(reference.token_embedding) / embedding_scale), -128, 127).astype(np.int8)
    gamma_fraction_bits = 13
    gamma_att = np.clip(np.rint(np.asarray(reference.rms_att) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    gamma_ffn = np.clip(np.rint(np.asarray(reference.rms_ffn) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    gamma_final = np.clip(np.rint(np.asarray(reference.rms_final) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    layers = []
    for layer in range(c.n_layers):
        layers.append({
            "wq": quantized_linear(reference.wq[layer], scales["norm"], scales["q"], bits, group_size, improved),
            "wk": quantized_linear(reference.wk[layer], scales["norm"], scales["k"], bits, group_size, improved),
            "wv": quantized_linear(reference.wv[layer], scales["norm"], scales["v"], bits, group_size, improved),
            "wo": quantized_linear(reference.wo[layer], scales["v"], scales["x"], bits, group_size, improved),
            "w1": quantized_linear(reference.w1[layer], scales["norm"], scales["h1"], bits, group_size, improved),
            "w2": quantized_linear(reference.w2[layer], scales["hidden"], scales["x"], bits, group_size, improved),
            "w3": quantized_linear(reference.w3[layer], scales["norm"], scales["h3"], bits, group_size, improved),
        })
    classifier = quantized_linear(reference.wcls, scales["norm"], scales["logits"], bits, group_size, improved)
    norm_mult, norm_shift = multiplier_shift(1.0 / ((1 << 15) * scales["norm"]))
    score_mult, score_shift = multiplier_shift(scales["q"] * scales["k"] / math.sqrt(c.head_size) * 64.0)
    swiglu_mult, swiglu_shift = multiplier_shift(scales["silu"] * scales["h3"] / scales["hidden"])
    one_mult, one_shift = multiplier_shift(1.0)
    return {"config": {"dim": c.dim, "hidden_dim": c.hidden_dim, "n_layers": c.n_layers,
                        "n_heads": c.n_heads, "n_kv_heads": c.n_kv_heads, "head_size": c.head_size,
                        "kv_dim": c.kv_dim, "vocab_size": c.vocab_size, "seq_len": c.seq_len},
            "embedding": embedding, "gamma_att": gamma_att, "gamma_ffn": gamma_ffn,
            "gamma_final": gamma_final, "layers": layers, "classifier": classifier,
            "norm_multiplier": norm_mult, "norm_shift": norm_shift,
            "gamma_fraction_bits": gamma_fraction_bits,
            "score_multiplier": score_mult, "score_shift": score_shift,
            "swiglu_multiplier": swiglu_mult, "swiglu_shift": swiglu_shift,
            "residual_multiplier": [one_mult, one_mult], "residual_shift": [one_shift, one_shift]}


def _adaptive_rows(values, base_scale):
    raw = np.rint(np.asarray(values, dtype=np.float64) / base_scale).astype(np.int64)
    peak = np.maximum(1, np.max(np.abs(raw), axis=-1)).astype(np.int32)
    code = np.clip(np.rint(raw * 128.0 / peak[..., None]), -128, 127).astype(np.int8)
    return code, peak


def export_runq_model(reference, calibration_scales, bits=8, group_size=32, improved=False):
    scales = {name: value / 256.0 for name, value in calibration_scales.items()}
    exported = export_model(reference, scales, bits, group_size, improved)
    code, scale_multiplier = _adaptive_rows(reference.token_embedding, scales["x"])
    exported["embedding"] = code
    exported["embedding_scale_multiplier"] = scale_multiplier
    exported["silu_index_multiplier"], exported["silu_index_shift"] = multiplier_shift(scales["h1"] * 64.0)
    exported["swiglu_multiplier"], exported["swiglu_shift"] = multiplier_shift(
        scales["h1"] * scales["h3"] / scales["hidden"])
    exported["rms_epsilon_x"] = max(1, int(round(1.0e-5 / (scales["x"] * scales["x"]))))
    exported["dynamic_base_scales"] = scales
    return exported


def accumulator_bounds(reference, bits, group_size):
    qmax = 127 if bits == 8 else 7
    weight_k = min(group_size, reference.config.dim)
    dims = {"qkv/wo/w1/w3/classifier": weight_k,
            "w2": min(group_size, reference.config.hidden_dim),
            }
    result = {name: {"K": int(k), "max_abs_a": 127, "max_abs_w": qmax,
                   "worst_case": int(k * 127 * qmax), "fits_int32": k * 127 * qmax <= (1 << 31) - 1}
            for name, k in dims.items()}
    score_k = reference.config.head_size
    result["attention_score"] = {"K": score_k, "max_abs_a": 127, "max_abs_w": 127,
                                 "worst_case": score_k * 127 * 127, "fits_int32": score_k * 127 * 127 <= (1 << 31) - 1}
    av_k = reference.config.seq_len
    result["attention_value"] = {"K": av_k, "max_abs_a": 32767, "max_abs_w": 127,
                                 "worst_case": av_k * 32767 * 127, "fits_int32": av_k * 32767 * 127 <= (1 << 31) - 1}
    return result
