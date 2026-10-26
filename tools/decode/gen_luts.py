"""Write numerics_v2 LUTs as $readmemh ROM images under rtl/decode/luts/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def main():
    luts = json.loads((ROOT / 'firmware/host/numerics_v2/luts/luts.json').read_text())
    out = ROOT / 'rtl/decode/luts'
    out.mkdir(parents=True, exist_ok=True)
    for name in ('sigmoid_q15', 'exp_q15', 'rope_cos_q15_h8', 'rope_sin_q15_h8', 'rope_cos_q15_h48', 'rope_sin_q15_h48'):
        values = np.asarray(luts[name]['values'], dtype=np.int64).reshape(-1)  # rope: position-major
        (out / f'{name}.mem').write_text(''.join(f'{int(v) & 0xFFFF:04x}\n' for v in values))
        print(name, values.size)


if __name__ == '__main__':
    main()
