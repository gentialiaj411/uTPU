"""Independent integer reference for the bounded tiny-transformer contract.

This module deliberately imports neither ISA encoder nor ISA simulator code.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple
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
        return self.vocab*self.D + self.T*self.D + self.L*(3*self.D*self.D + self.D*self.D + 2*self.D*self.Dff + 2*self.D) + self.D

def _rshift_rne(x: np.ndarray | int, shift: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.int64)
    if shift == 0: return a
    sign = np.where(a < 0, -1, 1); mag = np.abs(a); q = mag >> shift; rem = mag & ((1 << shift)-1); half = 1 << (shift-1)
    return sign * (q + ((rem > half) | ((rem == half) & ((q & 1) == 1))))

def vadd_rq(a: np.ndarray, ea: int, b: np.ndarray, eb: int, eo: int) -> Tuple[np.ndarray, int]:
    if eo < max(ea, eb): raise ValueError("eo must be >= both input exponents")
    total = _rshift_rne(np.asarray(a, np.int64), eo-ea) + _rshift_rne(np.asarray(b, np.int64), eo-eb)
    saturated = int(np.count_nonzero((total < -128) | (total > 127)))
    return np.clip(total, -128, 127).astype(np.int8), saturated

def softmax_u8(scores: np.ndarray, valid_count: int | None = None) -> np.ndarray:
    s = np.asarray(scores, dtype=np.int16).reshape(-1); n = len(s) if valid_count is None else int(valid_count)
    if not 0 < n <= len(s): raise ValueError("invalid valid_count")
    rel = np.clip(s[:n] - int(np.max(s[:n])), -128, 0); exp = EXP_Q15[(-rel).astype(np.int16)].astype(np.int64); total = int(exp.sum())
    out = np.zeros_like(s, dtype=np.uint8); out[:n] = ((255*exp + total//2)//total).astype(np.uint8)
    return out

def rmsnorm_i8(x: np.ndarray, gamma: np.ndarray, gamma_shift: int = 7) -> np.ndarray:
    v = int((np.sum(np.asarray(x, np.int64)**2) + len(x)//2)//len(x)) + 1; q = (v.bit_length()-1) & ~1
    m_q14 = min(65535, max(16384, int(((v << 14) + (1 << max(q-1, 0))) >> q)))
    r0 = int(RSQRT_Q14[min(191, (m_q14-16384) >> 8)]); t = int((m_q14*r0*r0 + (1<<27)) >> 28); r1 = int((r0*(3*(1<<14)-t) + (1<<14)) >> 15)
    y = _rshift_rne(np.asarray(x,np.int64)*np.asarray(gamma,np.int64)*r1, 14 + gamma_shift + q//2)
    return np.clip(y, -128, 127).astype(np.int8)

def _linear_i8(x: np.ndarray, weight: np.ndarray, shift: int = 7) -> np.ndarray:
    accum = np.matmul(np.asarray(x, np.int64), np.asarray(weight, np.int64).T)
    return np.clip(_rshift_rne(accum, shift), -128, 127).astype(np.int8)

def integer_decoder_logits(tokens: np.ndarray, state: dict[str, np.ndarray], cfg: TinyTransformerConfig) -> tuple[np.ndarray, dict[str, int]]:
    """Execute the bounded decoder with integer operations only.

    ``state`` is the exported INT8 state dictionary.  The function does not
    share lowering, encode/decode, or simulator code and is the Phase-1 oracle
    for later ISA tests.
    """
    tok = np.asarray(tokens, dtype=np.int64)
    if tok.ndim == 1: tok = tok[None, :]
    if tok.shape[1] != cfg.T: raise ValueError("tokens must have configured T")
    x = np.asarray(state['token.weight'], np.int8)[tok]
    pos = np.broadcast_to(np.asarray(state['pos.weight'], np.int8)[None, :cfg.T], x.shape)
    x, pos_sat = vadd_rq(x, 0, pos, 0, 1)
    counts = {'position_add': pos_sat, 'residual_attention': 0, 'residual_ffn': 0}
    for layer in range(cfg.L):
        p = f'blocks.{layer}.'; residual = x
        norm = np.stack([rmsnorm_i8(row, state[p+'n1.weight']) for batch in x for row in batch]).reshape(x.shape)
        qkv = _linear_i8(norm, state[p+'qkv.weight']).reshape(tok.shape[0], cfg.T, 3, cfg.H, cfg.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)); heads=[]
        for head in range(cfg.H):
            score = _rshift_rne(q[:, head].astype(np.int64) @ k[:, head].transpose(0,2,1).astype(np.int64), 4)
            score = np.clip(score, -128, 127).astype(np.int8); probs=np.stack([softmax_u8(score[b,i],i+1) for b in range(tok.shape[0]) for i in range(cfg.T)]).reshape(tok.shape[0],cfg.T,cfg.T)
            value = _rshift_rne(probs.astype(np.int64) @ v[:, head].astype(np.int64), 8); heads.append(np.clip(value,-128,127).astype(np.int8))
        attn=np.concatenate(heads,axis=-1); projected=_linear_i8(attn,state[p+'out.weight']); x,sat=vadd_rq(residual,0,projected,0,1);counts['residual_attention']+=sat
        residual=x; norm=np.stack([rmsnorm_i8(row,state[p+'n2.weight']) for batch in x for row in batch]).reshape(x.shape)
        hidden=_linear_i8(norm,state[p+'ff1.weight']); hidden=np.where(hidden<0,_rshift_rne(hidden,2),hidden).astype(np.int8)
        ff=_linear_i8(hidden,state[p+'ff2.weight']); x,sat=vadd_rq(residual,1,ff,1,2);counts['residual_ffn']+=sat
    norm=np.stack([rmsnorm_i8(row,state['final.weight']) for batch in x for row in batch]).reshape(x.shape)
    return _linear_i8(norm,state['token.weight']), counts
