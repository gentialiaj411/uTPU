"""Pack a numerics_v2 export into the byte image top_decode streams, plus its RTL parameters.

Layout (little-endian):
  CLS        classifier tiles (also the tied embedding codes)
  EMB_SCALE  per vocab row, per 32-wide group: multiplier u32, shift u8
  LAYERS     per layer: gamma_att (D x u16), wq, wk, wv, wo, gamma_ffn, w1, w3, w2; then gamma_final
Tensor tile format (rows padded to a multiple of 16):
  per tile: for each 32-wide weight group g: for each of the 16 rows: multiplier u32, shift u8
            then for each of the 16 rows: its codes (INT8: K bytes; INT4: K/2 bytes, even k in the low nibble)
The embedding row for token t is the classifier tile row: CLS + (t//16)*TILE + G*80 + (t%16)*KB.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

TILE_ROWS, GROUP = 16, 32


def tensor_bytes(rec):
    w = np.asarray(rec['weight'], dtype=np.int64)
    rows, k = w.shape
    groups = np.asarray(rec['groups'])
    assert all(int(groups[1, g] - groups[0, g]) == GROUP or g == groups.shape[1] - 1 for g in range(groups.shape[1]))
    assert all(int(groups[0, g]) == GROUP * g for g in range(groups.shape[1]))
    int4 = int(rec['qmax']) == 7
    mult, shift = np.asarray(rec['multiplier'], dtype=np.int64), np.asarray(rec['shift'], dtype=np.int64)
    assert np.all(mult >= 0) and np.all(mult < (1 << 32)) and np.all((shift >= 0) & (shift < 128))
    out = bytearray()
    for t0 in range(0, rows, TILE_ROWS):
        for g in range(groups.shape[1]):
            for r in range(t0, t0 + TILE_ROWS):
                m, s = (int(mult[r, g]), int(shift[r, g])) if r < rows else (0, 0)
                out += m.to_bytes(4, 'little') + bytes([s])
        for r in range(t0, t0 + TILE_ROWS):
            codes = w[r] if r < rows else np.zeros(k, dtype=np.int64)
            if int4:
                assert k % 2 == 0 and np.all((codes >= -8) & (codes <= 7))
                lo, hi = codes[0::2] & 15, codes[1::2] & 15
                out += bytes((lo | (hi << 4)).astype(np.uint8))
            else:
                out += bytes((codes & 0xFF).astype(np.uint8))
    return bytes(out), int4, k, groups.shape[1]


def pack(export, lut_dir_rel='rtl/decode/luts'):
    c = export['config']
    d, hdim, nl = int(c['dim']), int(c['hidden_dim']), int(c['n_layers'])
    img = bytearray()
    cls, cls4, _, gcls = tensor_bytes(export['classifier'])
    assert export.get('embedding_tied') and np.array_equal(np.asarray(export['embedding']), np.asarray(export['classifier']['weight']))
    base = {'CLS_BASE': 0}
    img += cls
    base['EMB_SCALE_BASE'] = len(img)
    em, es = np.asarray(export['embedding_group_multiplier']), np.asarray(export['embedding_group_shift'])
    assert int(export['embedding_group_size']) == GROUP
    for r in range(em.shape[0]):
        for g in range(em.shape[1]):
            img += int(em[r, g]).to_bytes(4, 'little') + bytes([int(es[r, g])])
    base['LAYERS_BASE'] = len(img)
    int4 = {}
    for li, layer in enumerate(export['layers']):
        img += np.asarray(export['gamma_att'][li], dtype='<u2').tobytes()
        for name in ('wq', 'wk', 'wv', 'wo'):
            b, f4, _, _ = tensor_bytes(layer[name]); img += b; int4[name] = f4
        img += np.asarray(export['gamma_ffn'][li], dtype='<u2').tobytes()
        for name in ('w1', 'w3', 'w2'):
            b, f4, _, _ = tensor_bytes(layer[name]); img += b; int4[name] = f4
    img += np.asarray(export['gamma_final'], dtype='<u2').tobytes()
    kb_cls = d // 2 if cls4 else d
    params = {
        'D': d, 'H': hdim, 'NL': nl, 'NH': int(c['n_heads']), 'NKV': int(c['n_kv_heads']), 'HS': int(c['head_size']),
        'VOCAB': int(c['vocab_size']), **base, 'IMAGE_BYTES': len(img),
        'CLS4': int(cls4), **{f'{k.upper()}4': int(v) for k, v in int4.items()},
        'CLS_TILE_BYTES': gcls * TILE_ROWS * 5 + TILE_ROWS * kb_cls, 'CLS_KB': kb_cls, 'CLS_G': gcls,
        'RECIP_D': int(round((1 << 32) / d)), 'GFB': int(export['gamma_fraction_bits']),
        'NORM_MULT': int(export['norm_multiplier']), 'NORM_SHIFT': int(export['norm_shift']), 'EPS': int(export['rms_epsilon_x']),
        'SILU_MULT': int(export['silu_index_multiplier']), 'SILU_SHIFT': int(export['silu_index_shift']),
        'SWIGLU_MULT': int(export['swiglu_multiplier']), 'SWIGLU_SHIFT': int(export['swiglu_shift']),
        'SCORE_MULT': int(export['score_multiplier']), 'SCORE_SHIFT': int(export['score_shift']),
    }
    return bytes(img), params


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import load_runtime_260k, BUILD
    exp, _ = load_runtime_260k()
    img, params = pack(exp)
    out = BUILD / 'decode_image_260k'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'image.bin').write_bytes(img)
    (out / 'params.json').write_text(json.dumps(params, indent=2) + '\n')
    print(json.dumps(params))


if __name__ == '__main__':
    main()
