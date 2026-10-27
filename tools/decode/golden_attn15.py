"""15M-shape attention golden: HS=48, 6 layers, 6 KV heads, KV4, positions up to 127.

Decodes SEQ tokens on the stories15M runtime with RunQIntegerLlama._score and runq_runtime._attention_value wrapped
(same capture as golden_attn.py); the final KV cache (rows are never rewritten) is dumped, and the head calls at
SAMPLE_POS positions are kept (every head of every layer at each). Random caches at the same shape go through the
runtime's own _score/_softmax/_to_base/_attention_value.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from harness import save_cases
from golden_attn import kv_dump
from run_decode_rtl import load_model
from firmware.host.numerics_v2 import runq_runtime as rr

SEQ = 128
SAMPLE_POS = (0, 1, 2, 15, 16, 17, 31, 32, 47, 48, 63, 64, 65, 90, 111, 112, 126, 127)
RAND = ((20, 150), (27, 60))


def capture(kv_bits=4):
    exp, rt = load_model('15m', kv_bits)
    c = exp['config']
    heads, layers = int(c['n_heads']), int(c['n_layers'])
    kv_mul = heads // int(c['n_kv_heads'])
    calls, pending = [], {}
    orig_score, orig_av = rr.RunQIntegerLlama._score, rr._attention_value

    def score(self, q, k8):
        out = orig_score(self, q, k8)
        if q.ndim == 2 and q.shape[0] == 1:
            q8 = rr._quantize(q[0])
            pending.update(q_code=q8.code.astype(int).tolist(), q_scale=int(q8.scale_multiplier[0]), score=out[0].astype(int).tolist())
        return out

    def av(probability, value):
        out = orig_av(probability, value)
        n = len(calls)
        layer, head = (n // heads) % layers, n % heads
        calls.append({'kind': f'real_kv{kv_bits}', 'layer': layer, 'head': head, 'kv_head': head // kv_mul,
                      'pos': len(rt.tokens) - 1, 'q_code': pending['q_code'], 'q_scale': pending['q_scale'],
                      'score': pending['score'], 'prob': np.asarray(probability).astype(int).tolist(), 'out': out.astype(int).tolist()})
        return out

    rr.RunQIntegerLlama._score, rr._attention_value = score, av
    rng = np.random.default_rng(15)
    tok = 1
    for _ in range(SEQ):
        rt.forward_argmax(tok, len(rt.tokens), 4096)
        tok = int(rng.integers(1, int(c['vocab_size'])))
    rr.RunQIntegerLlama._score, rr._attention_value = orig_score, orig_av
    keep = [x for x in calls if x['pos'] in SAMPLE_POS]
    return exp, rt, keep, len(calls)


def random_cases(exp, kv_bits, rng, count, v_bits):
    _, rt = load_model('15m', kv_bits)
    c = exp['config']
    hs, layers, kvh, seq = int(c['head_size']), int(c['n_layers']), int(c['n_kv_heads']), SEQ
    qmax = 127 if kv_bits == 8 else 7
    kc = rng.integers(-qmax, qmax + 1, size=(layers, seq, kvh * hs)).astype(np.int8)
    vc = rng.integers(-qmax, qmax + 1, size=(layers, seq, kvh * hs)).astype(np.int8)
    ks = rng.integers(1, 1 << int(rng.integers(8, 20)), size=(layers, seq, kvh)).astype(np.int32)
    vs = rng.integers(1, 1 << v_bits, size=(layers, seq, kvh)).astype(np.int32)
    if v_bits > 20:
        vc[:, :, :] = np.where(rng.random(vc.shape) < 0.5, qmax, -qmax)
    kc[:, :, :hs] = qmax; kc[0, 5] = -qmax
    cases = []
    while len(cases) < count:
        layer, kv_head, pos = int(rng.integers(0, layers)), int(rng.integers(0, kvh)), int(rng.integers(0, seq))
        if len(cases) % 5 == 0:
            pos = seq - 1 - int(rng.integers(0, 3))      # the deepest positions
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
    rng = np.random.default_rng(1515)
    exp, rt, calls, total = capture(4)
    groups = [{'kv_bits': 4, 'kv': kv_dump(rt), 'calls': calls}]
    n_real, n_rand = len(calls), 0
    for v_bits, count in RAND:
        kv, rand = random_cases(exp, 4, rng, count, v_bits)
        groups.append({'kv_bits': 4, 'kv': kv, 'calls': rand}); n_rand += len(rand)
    c = exp['config']
    meta = {'source': 'wrapped RunQIntegerLlama._score + runq_runtime._attention_value (stories15M decode path)',
            'real_cases': n_real, 'real_cases_captured': total, 'directed_random_cases': n_rand, 'head_size': int(c['head_size']),
            'n_layers': int(c['n_layers']), 'n_kv_heads': int(c['n_kv_heads']), 'max_seq': SEQ,
            'score_multiplier': int(rt.score_multiplier), 'score_shift': int(rt.score_shift)}
    _, sha = save_cases('attn15', meta, groups)
    print({**meta, 'sha256': sha, 'max_pos': max(x['pos'] for g in groups for x in g['calls'])})


if __name__ == '__main__':
    main()
