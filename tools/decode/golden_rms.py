"""Capture runq_runtime._rms_high vectors (stories260K decode) plus directed rounding-tie cases."""
from __future__ import annotations
import math
import numpy as np
from harness import load_runtime_260k, run_tokens, save_cases
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.fixed import INV7_Q32, INV127_Q32


def recip_q32(d):
    return {7: INV7_Q32, 127: INV127_Q32}.get(d, int(round((1 << 32) / d)))


def case(x, gamma, gfb, mult, shift, eps, out, kind):
    x = np.asarray(x, dtype=np.int64)
    ss = int(np.sum(x * x))
    mean = int(rr.div_const_rne(np.int64(ss), x.size)) + int(eps)
    return {'kind': kind, 'x': x.tolist(), 'gamma': np.asarray(gamma).astype(int).tolist(), 'gfb': int(gfb),
            'mult': int(mult), 'shift': int(shift), 'eps': int(eps), 'recip_d': recip_q32(x.size),
            'out': np.asarray(out).astype(int).tolist(), 'mean': mean, 'root': max(1, math.isqrt(mean << 30))}


def main():
    exp, runtime = load_runtime_260k()
    original = rr._rms_high
    cases = []

    def wrapped(value, gamma, gfb, multiplier, shift, epsilon):
        out = original(value, gamma, gfb, multiplier, shift, epsilon)
        cases.append(case(value, gamma, gfb, multiplier, shift, epsilon, out, 'real'))
        return out

    rr._rms_high = wrapped
    run_tokens(runtime)
    rr._rms_high = original
    n_real = len(cases)

    rng = np.random.default_rng(31)
    gfb, mult, shift, eps = exp['gamma_fraction_bits'], exp['norm_multiplier'], exp['norm_shift'], exp['rms_epsilon_x']

    def add(x, gamma, g=gfb, m=mult, s=shift, e=eps, kind='random'):
        x = np.asarray(x, dtype=np.int64)
        cases.append(case(x, gamma, g, m, s, e, original(x, np.asarray(gamma), g, m, s, e), kind))

    for i in range(120):
        d = 64 if i % 2 else 288
        mag = int(rng.integers(1, 1 << int(rng.integers(4, 25))))
        x = rng.integers(-mag, mag + 1, size=d)
        add(x, rng.integers(0, 1 << 16, size=d), m=int(rng.integers(1 << 28, (1 << 31) - 1)), s=int(rng.integers(24, 41)))
    for i in range(40):  # div_const_rne(ss, D) exactly on a tie
        d = 64 if i % 2 else 288
        x = rng.integers(-3000, 3001, size=d)
        for v in range(1, 100000):
            x[-1] = v
            if int(np.sum(x * x)) % d == d // 2:
                break
        add(x, rng.integers(0, 1 << 16, size=d), kind='div_tie')
    for i in range(40):  # mean+eps = 2^32 / 2^34: recip = 2^23 / 2^22 -> x*recip ties
        d = 64 if i % 2 else 288
        x = rng.integers(-60000, 60001, size=d) | (1 if i % 4 < 2 else 2)
        target = 1 << (32 if i % 4 < 2 else 34)
        m = int(rr.div_const_rne(np.int64(int(np.sum(x * x))), d))
        add(x, rng.integers(0, 1 << 16, size=d), e=target - m, kind='recip_tie')
    for i in range(40):  # gamma = 2^(gfb-1)*odd and multiplier = 2^(shift-1): later rshift_rne ties
        d = 64 if i % 2 else 288
        x = rng.integers(-5000, 5001, size=d)
        gamma = (2 * rng.integers(0, 8, size=d) + 1) << (gfb - 1)
        add(x, gamma, m=1 << 30, s=31, kind='gamma_out_tie')
    for i in range(10):  # zero vectors: root clamps via epsilon only
        d = 64 if i % 2 else 288
        add(np.zeros(d, dtype=np.int64), rng.integers(0, 1 << 16, size=d), kind='zero')

    meta = {'source': 'wrapped firmware.host.numerics_v2.runq_runtime._rms_high', 'real_cases': n_real,
            'directed_random_cases': len(cases) - n_real,
            'max_mean': max(c['mean'] for c in cases), 'max_root': max(c['root'] for c in cases)}
    path, sha = save_cases('rms', meta, cases)
    print({**meta, 'sha256': sha})


if __name__ == '__main__':
    main()
