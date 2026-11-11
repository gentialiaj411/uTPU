"""15M-shape GEMV tiles: real runq_runtime._linear_high calls from stories15M plus random/edge tiles at K=288 and K=768.

Real tiles come from the 15M runtime (models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz) decoding TOKENS; the layer
matmuls are captured in full (wq/wk/wv/wo/w2 INT8, w1/w3 INT4; K=288 except w2 K=768) and the tied classifier
(INT4, K=288) is captured for its first CLS_TILES tiles. Output is consumed by run_gemv_rtl.py --golden.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from harness import ROOT
from golden_gemv import save_record
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.runq_runtime import QValue
from run_decode_rtl import load_model

TOKENS = (1, 9038)
CLS_TILES = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'build/decode/golden_gemv15')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    _, runtime = load_model('15m', 4)
    original = rr._linear_high
    records, ks, cls_seen = [], {}, [0]

    def wrapped(value, record, output_begin=0, output_end=None):
        out = original(value, record, output_begin, output_end)
        begin = 0 if output_begin is None else output_begin
        end = record['weight'].shape[0] if output_end is None else output_end
        full = np.zeros(record['weight'].shape[0], dtype=np.int32)   # the runtime returns only [begin:end); save_record indexes absolute rows
        full[begin:end] = out
        cls = record['weight'].shape[0] > 4096   # tied classifier, however the runtime chunks it
        for b in range(begin, end, 16):
            if cls and cls_seen[0] >= CLS_TILES:
                break
            cls_seen[0] += cls
            records.append(save_record(a.output, len(records), value, record, full, b, min(b + 16, end)))
            ks[records[-1]['k'], records[-1]['qmax']] = ks.get((records[-1]['k'], records[-1]['qmax']), 0) + 1
        return out

    rr._linear_high = wrapped
    for t in TOKENS:
        cls_seen[0] = 0
        runtime.forward_argmax(t, len(runtime.tokens), 4096)
    rr._linear_high = original

    rng = np.random.default_rng(1515)
    random = []
    edge = np.array([-128, -127, -8, -7, -1, 0, 1, 7, 127], dtype=np.int8)
    i = 0
    while len(random) < 120:
        i += 1
        k = 288 if i % 2 else 768
        qmax = 127 if (i // 2) % 2 else 7
        rows = 16
        begins = list(range(0, k, 32))
        groups = np.asarray([begins, [min(b + 32, k) for b in begins]], dtype=np.int32)
        weight = rng.choice(edge, size=(rows, k)) if i % 4 else rng.integers(-128, 128, size=(rows, k)).astype(np.int8)
        if qmax == 7:
            weight = np.clip(weight, -8, 7).astype(np.int8)
        multip = rng.integers(1 << 16, (1 << 31) - 1, size=(rows, groups.shape[1]), dtype=np.int64)
        shifts = rng.integers(36, 48, size=(rows, groups.shape[1]), dtype=np.int64)   # lower shifts overflow INT32 at K>=288 and the tile is discarded after a slow reference run
        ac = (rng.choice(edge, size=k) if i % 5 else rng.integers(-128, 128, size=k)).astype(np.int8)
        scales = rng.integers(1, 1 << 17, size=((k + 63) // 64,), dtype=np.int64).astype(np.int32)
        rec = {'weight': weight, 'groups': groups, 'multiplier': multip, 'shift': shifts, 'qmax': qmax}
        q = QValue(ac, scales, 64, 127)
        try:
            out = original(q, rec)
        except OverflowError:
            continue
        random.append(save_record(a.output, 100000 + len(random), q, rec, out, 0, rows))
    manifest = {'model': 'stories15M', 'real_matmuls': len(records), 'random_matmuls': len(random),
                'real_output_elements': sum(r['output_elements'] for r in records),
                'random_output_elements': sum(r['output_elements'] for r in random),
                'real_k_qmax_tile_counts': {f'k{k}_qmax{q}': n for (k, q), n in sorted(ks.items())},
                'observed_max_widths': None, 'records': records, 'random_records': random,
                'source': 'wrapped firmware.host.numerics_v2.runq_runtime._linear_high (stories15M)'}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('real_matmuls', 'random_matmuls', 'real_output_elements', 'random_output_elements', 'real_k_qmax_tile_counts')}))


if __name__ == '__main__':
    main()
