"""Fault-inject decode RTL by source-text edits and confirm each runner reports mismatches.

Usage: python tools/decode/fault_inject.py <unit> [<unit> ...]
Results merge into bench/results/decode_fault_injection.json.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PKG = 'rtl/decode/decode_pkg.sv'

# unit -> (runner, {fault: (file, old, new)}); 'file' is the RTL or the shared package.
FAULTS = {
    'rms': ('tools/decode/run_rms_rtl.py', 'rtl/decode/rmsnorm.sv', {
        'mean_no_correction': ('rtl', "qf <= div_const_fix_resid(ss_mag > 64'h8000_0000, q0, r0, 10'(count));", "qf <= q0;"),
        'small_path_half_up': ('pkg', "(frac == 32'h8000_0000 && q[0])", "(frac == 32'h8000_0000)",
                               'equivalent: on an exact tie the half-up estimate is odd and the tie_odd correction restores RNE'),
        'recip_half_up': ('rtl', "== {1'b0, 56'(root_eff)}) && dv_ge)", "== {1'b0, 56'(root_eff)}))",
                          'equivalent: 2^54 mod root == root/2 has no solution, so div_rne(2^54, root) never ties'),
        'gamma_shift_off_by_one': ('rtl', "(step == 2'd1) ? 7'(gfb)", "(step == 2'd1) ? 7'(gfb) + 7'd1"),
        'seq_mul_drop_top_nibble': ('rtl', "mul_b <= 64'(recip); mul_start", "mul_b <= 64'(recip[33:0]); mul_start"),
        'shift_ties_away': ('pkg', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'isqrt_no_max1': ('rtl', "wire [46:0] root_eff = (root == 0) ? 47'd1 : root;", "wire [46:0] root_eff = root | 47'd1;"),
    }),
}


def main():
    path = ROOT / 'bench/results/decode_fault_injection.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for unit in sys.argv[1:]:
        runner, rtl, faults = FAULTS[unit]
        res = {}
        for name, (which, old, new, *why) in faults.items():
            src = ROOT / (PKG if which == 'pkg' else rtl)
            text = src.read_text()
            assert text.count(old) == 1, (unit, name)
            mutant = ROOT / 'build/decode' / f'fault_{unit}_{name}.sv'
            mutant.write_text(text.replace(old, new))
            out = ROOT / 'build/decode' / f'fault_{unit}_{name}.json'
            args = [sys.executable, str(ROOT / runner), '--output', str(out), '--pkg' if which == 'pkg' else '--rtl', str(mutant)]
            subprocess.run(args, check=True, capture_output=True)
            o = json.loads(out.read_text())
            res[name] = {'mismatches': o['mismatch_count'], 'real_mismatches': o['real_mismatch_count'], 'detected': o['mismatch_count'] > 0}
            if why:
                res[name]['equivalent_reason'] = why[0]
            res[name]['ok'] = res[name]['detected'] != bool(why)
            print(unit, name, res[name])
        results[unit] = res
    path.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
