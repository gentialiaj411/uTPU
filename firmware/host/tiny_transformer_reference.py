"""Independent scale-aware integer reference for the bounded transformer.

The reference deliberately imports neither ISA encoder nor simulator code. It
consumes the hash-checked export schema; no activation scale is implicit here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np

_LUTS = json.loads((Path(__file__).parent / "fixtures" / "tiny_transformer_integer_luts.json").read_text(encoding="utf-8"))
EXP_Q15 = np.asarray(_LUTS["exp_q15"], dtype=np.uint16)
RSQRT_Q14 = np.asarray(_LUTS["rsqrt_q14"], dtype=np.uint16)


@dataclass(frozen=True)
class TinyTransformerConfig:
    T: int = 32
    D: int = 48
    H: int = 3
    Dff: int = 192
    L: int = 1
    vocab: int = 65

    @property
    def head_dim(self) -> int:
        if self.D % self.H:
            raise ValueError("D must divide H")
        return self.D // self.H

    @property
    def parameter_count(self) -> int:
        return self.vocab * self.D + self.T * self.D + self.L * (3 * self.D * self.D + self.D * self.D + 2 * self.D * self.Dff + 2 * self.D) + self.D


def _rshift_rne(x: np.ndarray | int, shift: int | np.ndarray) -> np.ndarray:
    """Signed round-to-nearest, ties-to-even division by a power of two."""
    a = np.asarray(x, dtype=np.int64)
    s = np.asarray(shift, dtype=np.int64)
    safe = np.maximum(s, 1)
    mag = np.abs(a)
    q = np.right_shift(mag, safe)
    rem = mag - np.left_shift(q, safe)
    half = np.left_shift(np.ones_like(safe, dtype=np.int64), safe - 1)
    rounded = q + ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    signed = np.where(a < 0, -rounded, rounded)
    return np.where(s == 0, a, signed)


def _requant_accum(accum: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    multipliers = np.asarray(record["multipliers"], dtype=np.int64)
    shifts = np.asarray(record["right_shifts"], dtype=np.int64)
    if multipliers.shape != shifts.shape or multipliers.ndim != 1:
        raise ValueError("invalid exported multiplier/shift record")
    if multipliers.size not in (1, accum.shape[-1]):
        raise ValueError("exported requant channel count does not match output")
    return np.clip(_rshift_rne(np.asarray(accum, dtype=np.int64) * multipliers, shifts), -128, 127).astype(np.int8)


def vadd_rq(a: np.ndarray, ea: int, b: np.ndarray, eb: int, eo: int) -> Tuple[np.ndarray, int]:
    if eo < max(ea, eb):
        raise ValueError("eo must be >= both input exponents")
    total = _rshift_rne(np.asarray(a, np.int64), eo - ea) + _rshift_rne(np.asarray(b, np.int64), eo - eb)
    saturated = int(np.count_nonzero((total < -128) | (total > 127)))
    return np.clip(total, -128, 127).astype(np.int8), saturated


def softmax_u8(scores: np.ndarray, valid_count: int | None = None) -> np.ndarray:
    s = np.asarray(scores, dtype=np.int16).reshape(-1)
    n = len(s) if valid_count is None else int(valid_count)
    if not 0 < n <= len(s):
        raise ValueError("invalid valid_count")
    rel = np.clip(s[:n] - int(np.max(s[:n])), -128, 0)
    exp = EXP_Q15[(-rel).astype(np.int16)].astype(np.int64)
    total = int(exp.sum())
    out = np.zeros_like(s, dtype=np.uint8)
    out[:n] = ((255 * exp + total // 2) // total).astype(np.uint8)
    return out


def _divide_rne(value: int, denominator: int) -> int:
    """Nearest-even exact integer division for the RMS mean-square."""
    if denominator <= 0:
        raise ValueError("division denominator must be positive")
    sign = -1 if value < 0 else 1
    q, rem = divmod(abs(int(value)), int(denominator))
    twice = 2 * rem
    return sign * (q + int(twice > denominator or (twice == denominator and (q & 1))))


def rmsnorm_i8(x: np.ndarray, gamma: np.ndarray, gamma_shift: int) -> np.ndarray:
    """Contractual LUT+one-Newton RMSNorm; gamma_shift comes from export."""
    length = len(x)
    v = _divide_rne(int(np.sum(np.asarray(x, np.int64) ** 2)), length) + 1
    q = (v.bit_length() - 1) & ~1
    m_q14 = min(65535, max(16384, int(_rshift_rne(v << 14, q))))
    r0 = int(RSQRT_Q14[min(191, (m_q14 - 16384) >> 8)])
    t = int(_rshift_rne(m_q14 * r0 * r0, 28))
    r1 = int(_rshift_rne(r0 * (3 * (1 << 14) - t), 15))
    y = _rshift_rne(np.asarray(x, np.int64) * np.asarray(gamma, np.int64) * r1, 14 + int(gamma_shift) + q // 2)
    return np.clip(y, -128, 127).astype(np.int8)


def _linear_i8(x: np.ndarray, weight: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    return _requant_accum(np.asarray(x, np.int64) @ np.asarray(weight, np.int64).T, record)


def _required_schema(schema: dict[str, Any]) -> None:
    missing = {"activations", "linears", "norms", "residuals"} - set(schema)
    if missing:
        raise ValueError(f"export schema missing {sorted(missing)}")


def _integer_decoder(tokens: np.ndarray, state: dict[str, np.ndarray], schema: dict[str, Any], cfg: TinyTransformerConfig, capture_trace: bool) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    _required_schema(schema)
    acts, linears, norms, residuals = (schema["activations"], schema["linears"], schema["norms"], schema["residuals"])
    tok = np.asarray(tokens, dtype=np.int64)
    if tok.ndim == 1:
        tok = tok[None, :]
    if tok.shape[1] != cfg.T:
        raise ValueError("tokens must have configured T")
    x = np.asarray(state["token.weight"], np.int8)[tok]
    pos = np.broadcast_to(np.asarray(state["pos.weight"], np.int8)[None, :cfg.T], x.shape)
    trace: dict[str, np.ndarray] = {}
    pr = residuals["pos_add"]
    x, pos_sat = vadd_rq(x, int(pr["ea"]), pos, int(pr["eb"]), int(pr["eo"]))
    if capture_trace:
        trace["pos_add"] = x.copy()
    counts = {"position_add": pos_sat, "residual_attention": 0, "residual_ffn": 0}
    for layer in range(cfg.L):
        p = f"blocks.{layer}."
        residual = x
        n1 = norms["n1"]
        norm = np.stack([rmsnorm_i8(row, state[p + "n1.weight"], int(n1["gamma_shift"])) for batch in x for row in batch]).reshape(x.shape)
        if capture_trace:
            trace["rmsnorm1"] = norm.copy()
        qkv = _linear_i8(norm, state[p + "qkv.weight"], linears["qkv"]).reshape(tok.shape[0], cfg.T, 3, cfg.H, cfg.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        if capture_trace:
            trace["q"], trace["k"], trace["v"] = q.copy(), k.copy(), v.copy()
        all_scores, all_probs, heads = [], [], []
        for head in range(cfg.H):
            score_acc = q[:, head].astype(np.int64) @ k[:, head].transpose(0, 2, 1).astype(np.int64)
            # The exact 1/4 head-dimension scale is folded into this record.
            score = _requant_accum(score_acc, linears["scores"])
            all_scores.append(score)
            probs = np.stack([softmax_u8(score[b, i], i + 1) for b in range(tok.shape[0]) for i in range(cfg.T)]).reshape(tok.shape[0], cfg.T, cfg.T)
            all_probs.append(probs)
            heads.append(_requant_accum(probs.astype(np.int64) @ v[:, head].astype(np.int64), linears["av"]))
        attn = np.concatenate(heads, axis=-1)
        if capture_trace:
            trace["scores"] = np.stack(all_scores, axis=1)
            trace["softmax_probs"] = np.stack(all_probs, axis=1)
            trace["av"] = attn.copy()
        projected = _linear_i8(attn, state[p + "out.weight"], linears["out"])
        if capture_trace:
            trace["out_proj"] = projected.copy()
        rr = residuals["residual1"]
        x, sat = vadd_rq(residual, int(rr["ea"]), projected, int(rr["eb"]), int(rr["eo"]))
        counts["residual_attention"] += sat
        if capture_trace:
            trace["residual1"] = x.copy()
        residual = x
        n2 = norms["n2"]
        norm = np.stack([rmsnorm_i8(row, state[p + "n2.weight"], int(n2["gamma_shift"])) for batch in x for row in batch]).reshape(x.shape)
        if capture_trace:
            trace["rmsnorm2"] = norm.copy()
        hidden = _linear_i8(norm, state[p + "ff1.weight"], linears["ff1"])
        if capture_trace:
            trace["ffn1"] = hidden.copy()
        hidden = np.where(hidden < 0, _rshift_rne(hidden, 2), hidden).astype(np.int8)
        if capture_trace:
            trace["leaky_relu"] = hidden.copy()
        ff = _linear_i8(hidden, state[p + "ff2.weight"], linears["ff2"])
        if capture_trace:
            trace["ffn2"] = ff.copy()
        rr = residuals["residual2"]
        x, sat = vadd_rq(residual, int(rr["ea"]), ff, int(rr["eb"]), int(rr["eo"]))
        counts["residual_ffn"] += sat
        if capture_trace:
            trace["residual2"] = x.copy()
    final = norms["final"]
    norm = np.stack([rmsnorm_i8(row, state["final.weight"], int(final["gamma_shift"])) for batch in x for row in batch]).reshape(x.shape)
    if capture_trace:
        trace["final_norm"] = norm.copy()
    logits = _linear_i8(norm, state["token.weight"], linears["logits"])
    if capture_trace:
        trace["logits"] = logits.copy()
    return logits, counts, trace


def integer_decoder_logits(tokens: np.ndarray, state: dict[str, np.ndarray], schema: dict[str, Any], cfg: TinyTransformerConfig) -> tuple[np.ndarray, dict[str, int]]:
    logits, counts, _ = _integer_decoder(tokens, state, schema, cfg, capture_trace=False)
    return logits, counts


def integer_decoder_trace(tokens: np.ndarray, state: dict[str, np.ndarray], schema: dict[str, Any], cfg: TinyTransformerConfig) -> dict[str, np.ndarray]:
    _, _, trace = _integer_decoder(tokens, state, schema, cfg, capture_trace=True)
    return trace
