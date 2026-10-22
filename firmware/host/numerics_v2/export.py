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


def _clip_for_mse(values, qmax, hessian=None):
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return 1.0
    best = (float("inf"), peak)
    for fraction in np.linspace(0.55, 1.0, 46):
        clip = peak * float(fraction)
        scale = clip / qmax
        code = np.clip(np.rint(values / scale), -qmax, qmax)
        weight = 1.0 if hessian is None else np.asarray(hessian, dtype=np.float64)
        error = float(np.sum(weight * (values - code * scale) ** 2))
        if error < best[0]:
            best = (error, clip)
    return best[1]


def quantized_linear(weight, input_scale, output_scale, bits, group_size, improved=False,
                     hessian=None, gptq=False, embedding_scale=None):
    value = np.asarray(weight, dtype=np.float32)
    out, width = value.shape
    qmax = 127 if bits == 8 else 7
    size = group_size
    ranges = [(start, min(start + size, width)) for start in range(0, width, size)]
    codes = np.empty_like(value, dtype=np.int8)
    multipliers = np.empty((out, len(ranges)), dtype=np.int64)
    shifts = np.empty((out, len(ranges)), dtype=np.int64)
    weight_scale_multipliers = np.empty((out, len(ranges)), dtype=np.int64)
    weight_scale_shifts = np.empty((out, len(ranges)), dtype=np.int64)
    embedding_scale_multipliers = np.empty((out, len(ranges)), dtype=np.int64)
    embedding_scale_shifts = np.empty((out, len(ranges)), dtype=np.int64)
    for row in range(out):
        for group, (begin, end) in enumerate(ranges):
            block = value[row, begin:end]
            hdiag = None if hessian is None else np.asarray(hessian[begin:end], dtype=np.float64)
            clip = (_clip_for_mse(block, qmax, hdiag) if (improved or gptq) and bits == 4
                    else float(np.max(np.abs(block))))
            scale = clip / qmax if clip else 1.0
            codes[row, begin:end] = np.clip(np.rint(block / scale), -qmax, qmax).astype(np.int8)
            multipliers[row, group], shifts[row, group] = multiplier_shift(input_scale * scale / output_scale)
            weight_scale_multipliers[row, group], weight_scale_shifts[row, group] = multiplier_shift(scale / input_scale)
            if embedding_scale is not None:
                embedding_scale_multipliers[row, group], embedding_scale_shifts[row, group] = multiplier_shift(scale / embedding_scale)
    result = {"weight": codes, "multiplier": multipliers, "shift": shifts,
            "weight_scale_multiplier": weight_scale_multipliers,
            "weight_scale_shift": weight_scale_shifts,
            "groups": np.asarray([[a for a, _ in ranges], [b for _, b in ranges]], dtype=np.int32),
            "qmax": qmax}
    if embedding_scale is not None:
        result["embedding_scale_multiplier"] = embedding_scale_multipliers
        result["embedding_scale_shift"] = embedding_scale_shifts
    return result


def export_model(reference, scales, bits=8, group_size=32, improved=False,
                 hessian_diag=None, mixed_precision=None, gptq=False):
    c = reference.config
    embedding_scale = scales["x"]
    embedding = np.clip(np.rint(np.asarray(reference.token_embedding) / embedding_scale), -128, 127).astype(np.int8)
    gamma_fraction_bits = 12
    gamma_att = np.clip(np.rint(np.asarray(reference.rms_att) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    gamma_ffn = np.clip(np.rint(np.asarray(reference.rms_ffn) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    gamma_final = np.clip(np.rint(np.asarray(reference.rms_final) * (1 << gamma_fraction_bits)), 0, 65535).astype(np.uint16)
    layers = []
    mixed_precision = mixed_precision or {}
    hessian_diag = hessian_diag or {}
    def tensor_bits(name):
        return int(mixed_precision.get(name, bits))
    def tensor_hessian(name):
        return hessian_diag.get(name)
    for layer in range(c.n_layers):
        layers.append({
            "wq": quantized_linear(reference.wq[layer], scales["norm"], scales["q"], tensor_bits("qkv"), group_size, improved, tensor_hessian("norm"), gptq),
            "wk": quantized_linear(reference.wk[layer], scales["norm"], scales["k"], tensor_bits("qkv"), group_size, improved, tensor_hessian("norm"), gptq),
            "wv": quantized_linear(reference.wv[layer], scales["norm"], scales["v"], tensor_bits("qkv"), group_size, improved, tensor_hessian("norm"), gptq),
            "wo": quantized_linear(reference.wo[layer], scales["v"], scales["x"], tensor_bits("wo"), group_size, improved, tensor_hessian("av"), gptq),
            "w1": quantized_linear(reference.w1[layer], scales["norm"], scales["h1"], tensor_bits("w13"), group_size, improved, tensor_hessian("norm"), gptq),
            "w2": quantized_linear(reference.w2[layer], scales["hidden"], scales["x"], tensor_bits("w2"), group_size, improved, tensor_hessian("hidden"), gptq),
            "w3": quantized_linear(reference.w3[layer], scales["norm"], scales["h3"], tensor_bits("w13"), group_size, improved, tensor_hessian("norm"), gptq),
        })
    classifier = quantized_linear(reference.wcls, scales["norm"], scales["logits"], tensor_bits("classifier"), group_size, improved, tensor_hessian("norm"), gptq,
                                  embedding_scale=scales["x"] if c.shared_classifier else None)
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


def export_runq_model(reference, calibration_scales, bits=8, group_size=32, improved=False,
                      hessian_diag=None, mixed_precision=None, gptq=False):
    scales = {name: value / 256.0 for name, value in calibration_scales.items()}
    exported = export_model(reference, scales, bits, group_size, improved,
                            hessian_diag=hessian_diag, mixed_precision=mixed_precision, gptq=gptq)
    if reference.config.shared_classifier:
        # The checkpoint ties wcls to token_embedding.  Reuse classifier codes and
        # retain a second scale-pair plane for embedding lookup; no duplicate table.
        exported["embedding"] = exported["classifier"]["weight"]
        exported["embedding_group_multiplier"] = exported["classifier"]["embedding_scale_multiplier"]
        exported["embedding_group_shift"] = exported["classifier"]["embedding_scale_shift"]
        exported["embedding_group_size"] = int(group_size)
        exported["embedding_tied"] = True
    else:
        code, scale_multiplier = _adaptive_rows(reference.token_embedding, scales["x"])
        exported["embedding"] = code
        exported["embedding_scale_multiplier"] = scale_multiplier
        exported["embedding_scale_shift"] = 7
        exported["embedding_tied"] = False
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
