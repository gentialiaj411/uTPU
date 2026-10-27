"""Write numerics_v2 LUTs as $readmemh ROM images under rtl/decode/luts/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ROPE_POSITIONS_H48 = 128   # = 15M MAX_SEQ


def main():
    luts = json.loads((ROOT / 'firmware/host/numerics_v2/luts/luts.json').read_text())
    out = ROOT / 'rtl/decode/luts'
    out.mkdir(parents=True, exist_ok=True)
    def pair(c, s):   # {cos, sin} per line for the single 32-bit-wide RoPE ROM
        return ''.join(f'{int(a) & 0xFFFF:04x}{int(b) & 0xFFFF:04x}\n' for a, b in zip(c, s))
    for h, tag in ((8, 'h8'), (48, 'h48')):
        c = np.asarray(luts[f'rope_cos_q15_{tag}']['values'], dtype=np.int64).reshape(-1)
        s = np.asarray(luts[f'rope_sin_q15_{tag}']['values'], dtype=np.int64).reshape(-1)
        (out / f'rope_cs_q15_{tag}.mem').write_text(pair(c, s))
        if h == 48:
            n = ROPE_POSITIONS_H48 * (h // 2)
            (out / f'rope_cs_q15_{tag}_p{ROPE_POSITIONS_H48}.mem').write_text(pair(c[:n], s[:n]))
    for name in ('sigmoid_q15', 'exp_q15', 'rope_cos_q15_h8', 'rope_sin_q15_h8', 'rope_cos_q15_h48', 'rope_sin_q15_h48'):
        values = np.asarray(luts[name]['values'], dtype=np.int64).reshape(-1)  # rope: position-major
        (out / f'{name}.mem').write_text(''.join(f'{int(v) & 0xFFFF:04x}\n' for v in values))
        print(name, values.size)
        if name.startswith('rope') and name.endswith('h48'):   # 15M, MAX_SEQ 128: rows past the context window are never read
            half = 48 // 2
            trunc = values[:ROPE_POSITIONS_H48 * half]
            (out / f'{name}_p{ROPE_POSITIONS_H48}.mem').write_text(''.join(f'{int(v) & 0xFFFF:04x}\n' for v in trunc))
            print(f'{name}_p{ROPE_POSITIONS_H48}', trunc.size)


if __name__ == '__main__':
    main()
