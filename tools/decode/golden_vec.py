"""Capture element-wise decode ops for vec_alu: RoPE, SwiGLU hidden, residual add, embedding scaling.

Real cases wrap the runtime's own _rope_high, _silu_high/_hidden_high and _embedding_high during a
stories260K decode; directed cases call the same functions (or fixed.rshift_rne for the embedding
multiply) with configurations chosen to land on exact rounding ties and range edges.
"""
from __future__ import annotations
import copy
import numpy as np
from harness import load_runtime_260k, run_tokens, save_cases
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.fixed import rshift_rne

OP_ROPE, OP_SWIGLU, OP_ADD, OP_MULSHIFT = 0, 1, 2, 3


def i32(v):
    return [int(x) for x in np.asarray(v, dtype=np.int64).reshape(-1)]


def main():
    exp, rt = load_runtime_260k()
    cls = rr.RunQIntegerLlama
    half = int(exp['config']['head_size']) // 2
    base_cfg = {'silu_mult': rt.silu_index_multiplier, 'silu_shift': rt.silu_index_shift,
                'swiglu_mult': rt.swiglu_multiplier, 'swiglu_shift': rt.swiglu_shift}
    cases = []

    def rope_case(x, pos, out, kind):
        pairs = np.asarray(x, dtype=np.int64).reshape(-1, 2)
        o = np.asarray(out, dtype=np.int64).reshape(-1, 2)
        cases.append({'kind': kind, 'op': OP_ROPE, 'cfg': {**base_cfg, 'rope_base': int(pos) * half, 'rope_half': half},
                      'beats': [[int(p[0]), int(p[1]), 0, 0] for p in pairs], 'out': [[int(r[0]), int(r[1])] for r in o]})

    def swiglu_case(h1, h3, out, cfg, kind):
        cases.append({'kind': kind, 'op': OP_SWIGLU, 'cfg': {**cfg, 'rope_base': 0, 'rope_half': half},
                      'beats': [[a, b, 0, 0] for a, b in zip(i32(h1), i32(h3))], 'out': [[v, 0] for v in i32(out)]})

    def mulshift_case(a, m, s, kind):
        a = np.asarray(a, dtype=np.int64); m = np.asarray(m, dtype=np.int64); s = np.asarray(s, dtype=np.int64)
        out = rshift_rne(a * m, s).astype(np.int32)
        cases.append({'kind': kind, 'op': OP_MULSHIFT, 'cfg': {**base_cfg, 'rope_base': 0, 'rope_half': half},
                      'beats': [[int(x), 0, int(y), int(z)] for x, y, z in zip(a, m, s)], 'out': [[v, 0] for v in i32(out)]})

    # ---- real captures -------------------------------------------------------------------------
    orig_rope, orig_silu, orig_hidden, orig_emb = cls._rope_high, cls._silu_high, cls._hidden_high, cls._embedding_high
    pending = {}

    def rope(self, value, positions):
        out = orig_rope(self, value, positions)
        rope_case(value, positions, out, 'real')
        return out

    def silu(self, value):
        out = orig_silu(self, value)
        pending['h1'] = np.asarray(value).copy()
        return out

    def hidden(self, s, h3):
        out = orig_hidden(self, s, h3)
        swiglu_case(pending.pop('h1'), h3, out, base_cfg, 'real')
        return out

    def emb(self, ids):
        out = orig_emb(self, ids)
        code = self.embedding[ids].astype(np.int64)
        g = self.embedding_group_size
        groups = np.arange(code.size) // g
        a, m, s = code, self.embedding_group_multiplier[ids, groups], self.embedding_group_shift[ids, groups]
        assert np.array_equal(rshift_rne(a * m, s).astype(np.int32), out)
        mulshift_case(a, m, s, 'real')
        return out

    cls._rope_high, cls._silu_high, cls._hidden_high, cls._embedding_high = rope, silu, hidden, emb
    run_tokens(rt)
    cls._rope_high, cls._silu_high, cls._hidden_high, cls._embedding_high = orig_rope, orig_silu, orig_hidden, orig_emb
    n_real = len(cases)

    rng = np.random.default_rng(55)
    # ---- RoPE: random positions/magnitudes and exact ties (x1 = 0, x0*c == 2^14 mod 2^15) ----------
    for i in range(60):
        pos = int(rng.integers(0, 512)); mag = 1 << int(rng.integers(4, 31))
        x = rng.integers(-mag, mag, size=2 * half * 4)
        rope_case(x, pos, rt._rope_high(x, pos), 'random')
    for i in range(60):
        pos = int(rng.integers(0, 512))
        x = rng.integers(-(1 << 20), 1 << 20, size=2 * half).astype(np.int64)
        for j in range(half):
            c = int(rt.rope_cos[pos, j]); s = int(rt.rope_sin[pos, j])
            v = c if (i % 2 == 0) else s
            if v % 2:
                x0 = ((1 << 14) * pow(v % (1 << 15), -1, 1 << 15)) % (1 << 15)
                x[2 * j] = x0 + (1 << 15) * int(rng.integers(-500, 500)); x[2 * j + 1] = 0
        rope_case(x, pos, rt._rope_high(x, pos), 'rope_tie')
    # ---- SwiGLU: random magnitudes, sigmoid table edges, and tie-forcing configurations ------------
    def run_swiglu(h1, h3, cfg, kind):
        r = copy.copy(rt)
        r.silu_index_multiplier, r.silu_index_shift = cfg['silu_mult'], cfg['silu_shift']
        r.swiglu_multiplier, r.swiglu_shift = cfg['swiglu_mult'], cfg['swiglu_shift']
        h1 = np.asarray(h1, dtype=np.int32); h3 = np.asarray(h3, dtype=np.int32)
        swiglu_case(h1, h3, r._hidden_high(r._silu_high(h1), h3), cfg, kind)

    for i in range(40):
        mag = 1 << int(rng.integers(4, 31))
        run_swiglu(rng.integers(-mag, mag, 64), rng.integers(-mag, mag, 64), base_cfg, 'random')
    tie_cfg = {'silu_mult': 1 << 30, 'silu_shift': 31, 'swiglu_mult': 1 << 16, 'swiglu_shift': 17}
    for i in range(30):
        h1 = rng.integers(-1100, 1100, 64)                                    # idx = rne(h1/2): ties, +-512 edges
        h1[:8] = [1024, 1025, 1026, -1024, -1025, -1026, 1023, -1023]
        if i % 2:
            h1[8:] = (1 << 14) + (1 << 15) * rng.integers(1, 3000, 56)         # sig = 32767: h1*sig ties at >>15
        run_swiglu(h1, rng.integers(-(1 << 15), 1 << 15, 64) | 1, tie_cfg, 'swiglu_tie')
    # ---- residual add: wrap-around ----------------------------------------------------------------
    for i in range(20):
        a = rng.integers(-(1 << 31), 1 << 31, 64); b = rng.integers(-(1 << 31), 1 << 31, 64)
        out = (a.astype(np.int64) + b).astype(np.int32)                       # x.astype(int64)+y -> astype(int32)
        cases.append({'kind': 'random', 'op': OP_ADD, 'cfg': {**base_cfg, 'rope_base': 0, 'rope_half': half},
                      'beats': [[int(x), int(y), 0, 0] for x, y in zip(a, b)], 'out': [[v, 0] for v in i32(out)]})
    # ---- embedding multiply-shift: random and ties (m = odd * 2^(s-1)) -----------------------------
    for i in range(30):
        s = rng.integers(1, 40, 64) if i % 2 else rng.integers(1, 29, 64)
        m = rng.integers(1, (1 << 31) - 1, 64) if i % 2 else ((2 * rng.integers(0, 4, 64) + 1) << (np.minimum(s, 28) - 1))
        mulshift_case(rng.integers(-128, 128, 64), m, s, 'random' if i % 2 else 'mulshift_tie')

    meta = {'source': 'wrapped RunQIntegerLlama._rope_high/_silu_high/_hidden_high/_embedding_high',
            'real_cases': n_real, 'directed_random_cases': len(cases) - n_real}
    _, sha = save_cases('vec', meta, cases)
    print({**meta, 'beats': sum(len(c['beats']) for c in cases), 'sha256': sha})


if __name__ == '__main__':
    main()
