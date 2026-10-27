"""Capture per-head decode attention (score -> softmax -> attention-value) from runq_runtime.

For each KV width (8 and 4) a stories260K runtime decodes TOKENS tokens. Wrappers around
RunQIntegerLlama._score and runq_runtime._attention_value record every head call: the quantized q
(the runtime's own _quantize(q), exactly as _score computes it), layer/head/position, and the head
output. The final KV cache (rows are never rewritten) is dumped so the RTL testbench can preload it.
Random cases call the same runtime functions on random caches.
"""
from __future__ import annotations
import numpy as np
from harness import load_runtime_260k, save_cases
from firmware.host.numerics_v2 import runq_runtime as rr

TOKENS = [1, 403, 260, 87, 33, 499, 12, 300, 77, 150] * 4   # 40 decode steps


def kv_dump(rt):
    return {'key_code': rt.key_code.astype(int).tolist(), 'value_code': rt.value_code.astype(int).tolist(),
            'key_scale': rt.key_scale.astype(int).tolist(), 'value_scale': rt.value_scale.astype(int).tolist()}


def capture(kv_bits):
    exp, rt = load_runtime_260k(kv_bits)
    c = exp['config']
    heads, layers = int(c['n_heads']), int(c['n_layers'])
    kv_mul = heads // int(c['n_kv_heads'])
    calls, pending = [], {}
    orig_score, orig_av = rr.RunQIntegerLlama._score, rr._attention_value

    def score(self, q, k8):
        out = orig_score(self, q, k8)
        if q.ndim == 2 and q.shape[0] == 1:
            q8 = rr._quantize(q[0])
            n = len(calls)
            pending.update(q_code=q8.code.astype(int).tolist(), q_scale=int(q8.scale_multiplier[0]),
                           score=out[0].astype(int).tolist(), n=n)
        return out

    def av(probability, value):
        out = orig_av(probability, value)
        n = len(calls)
        layer, head = (n // heads) % layers, n % heads
        calls.append({'kind': f'real_kv{kv_bits}', 'layer': layer, 'head': head, 'kv_head': head // kv_mul,
                      'pos': len(rt.tokens) - 1, 'q_code': pending['q_code'], 'q_scale': pending['q_scale'],
                      'score': pending['score'], 'prob': np.asarray(probability).astype(int).tolist(),
                      'out': out.astype(int).tolist()})
        return out

    rr.RunQIntegerLlama._score, rr._attention_value = score, av
    for t in TOKENS:
        rt.forward_argmax(t, len(rt.tokens), 4096)
    rr.RunQIntegerLlama._score, rr._attention_value = orig_score, orig_av
    return exp, rt, calls


def random_cases(exp, kv_bits, rng, count, v_bits=20):
    """Random caches through the runtime's own score/softmax/attention-value functions."""
    _, rt = load_runtime_260k(kv_bits)
    c = exp['config']
    hs, layers, kvh, seq = int(c['head_size']), int(c['n_layers']), int(c['n_kv_heads']), 64
    qmax = 127 if kv_bits == 8 else 7
    kc = rng.integers(-qmax, qmax + 1, size=(layers, seq, kvh * hs)).astype(np.int8)
    vc = rng.integers(-qmax, qmax + 1, size=(layers, seq, kvh * hs)).astype(np.int8)
    ks = rng.integers(1, 1 << int(rng.integers(8, 20)), size=(layers, seq, kvh)).astype(np.int32)
    vs = rng.integers(1, 1 << v_bits, size=(layers, seq, kvh)).astype(np.int32)
    if v_bits > 20:
        vc[:, :, :] = np.where(rng.random(vc.shape) < 0.5, qmax, -qmax)      # saturated codes: block peaks near 2^v_bits
    kc[:, :, :hs] = qmax; kc[0, 5] = -qmax                                    # saturated codes
    cases = []
    while len(cases) < count:
        layer, kv_head, pos = int(rng.integers(0, layers)), int(rng.integers(0, kvh)), int(rng.integers(0, seq))
        mag = 1 << int(rng.integers(2, 18))
        q = rng.integers(-mag, mag + 1, size=hs).astype(np.int64)
        b = kv_head * hs
        key = rr.QValue(kc[layer, :pos + 1, b:b + hs], ks[layer, :pos + 1, kv_head, None], hs, qmax)
        value = rr.QValue(vc[layer, :pos + 1, b:b + hs], vs[layer, :pos + 1, kv_head, None], hs, qmax)
        try:
            score = rt._score(q[None, :], key)[0]
            prob = rt._softmax(score)
            out = rr._attention_value(prob, rr._to_base(value))
        except (OverflowError, AssertionError):
            continue
        q8 = rr._quantize(q)
        cases.append({'kind': f'random_kv{kv_bits}_v{v_bits}', 'layer': layer, 'head': kv_head, 'kv_head': kv_head, 'pos': pos,
                      'q_code': q8.code.astype(int).tolist(), 'q_scale': int(q8.scale_multiplier[0]),
                      'score': score.astype(int).tolist(), 'prob': prob.astype(int).tolist(), 'out': out.astype(int).tolist()})
    return {'key_code': kc.astype(int).tolist(), 'value_code': vc.astype(int).tolist(),
            'key_scale': ks.astype(int).tolist(), 'value_scale': vs.astype(int).tolist()}, cases


def main():
    rng = np.random.default_rng(66)
    groups, n_real, n_rand = [], 0, 0
    for kv_bits in (8, 4):
        exp, rt, calls = capture(kv_bits)
        groups.append({'kv_bits': kv_bits, 'kv': kv_dump(rt), 'calls': calls}); n_real += len(calls)
        for v_bits, count in ((20, 150), (27, 60)):     # 2^27 V scales reach the -128 code clamp
            kv, rand = random_cases(exp, kv_bits, rng, count, v_bits)
            groups.append({'kv_bits': kv_bits, 'kv': kv, 'calls': rand}); n_rand += len(rand)
    c = exp['config']
    meta = {'source': 'wrapped RunQIntegerLlama._score + runq_runtime._attention_value (decode path)',
            'real_cases': n_real, 'directed_random_cases': n_rand, 'head_size': int(c['head_size']),
            'n_layers': int(c['n_layers']), 'n_kv_heads': int(c['n_kv_heads']),
            'score_multiplier': int(rt.score_multiplier), 'score_shift': int(rt.score_shift)}
    _, sha = save_cases('attn', meta, groups)
    print({**meta, 'sha256': sha, 'max_pos': max(x['pos'] for g in groups for x in g['calls'])})


if __name__ == '__main__':
    main()
