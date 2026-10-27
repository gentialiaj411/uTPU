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
    'vec': ('tools/decode/run_vec_rtl.py', 'rtl/decode/vec_alu.sv', {
        'rope_sign': ('rtl', "r0 = 32'(rshift_rne(t0 - t1, 7'd15));", "r0 = 32'(rshift_rne(t0 + t1, 7'd15));"),
        'rope_no_pair_wrap': ('rtl', "(pair_i + 1'b1 == rope_half) ? '0 :", "1'b0 ? '0 :"),
        'sigmoid_edge': ('rtl', "(idx_r > 64'sd512) ? 16'd32767", "(idx_r >= 64'sd512) ? 16'd32767"),
        'swiglu_drop_high_word': ('rtl', "wire signed [63:0] low64 = t2 + (t3 <<< 32);", "wire signed [63:0] low64 = t2;"),
        'shift_ties_away': ('pkg', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'add_saturate': ('rtl', "OP_ADD: r0 = a_r + b_r;", "OP_ADD: r0 = (33'(a_r) + 33'(b_r) > 33'sd2147483647) ? 32'sd2147483647 : a_r + b_r;"),
    }),
    'attn': ('tools/decode/run_attn_rtl.py', 'rtl/decode/attention.sv', {
        'exp_clip_1023': ('rtl', "(d > 34'sd1024) ? 11'd1024 : 11'(d)", "(d > 34'sd1023) ? 11'd1023 : 11'(d)",
                          'equivalent: exp_q15 is 0 for every index >= 710, so clipping at 1023 or 1024 reads 0'),
        'exp_clip_512': ('rtl', "(d > 34'sd1024) ? 11'd1024 : 11'(d)", "(d > 34'sd512) ? 11'd512 : 11'(d)"),
        'kv4_score_div127': ('rtl', "dp_d127 <= !kv4; state <= SC_DWAIT;", "dp_d127 <= 1'b1; state <= SC_DWAIT;"),
        'av_clip_127': ('rtl', "(r < -64'sd128) ? -32'sd128", "(r < -64'sd127) ? -32'sd127"),
        'softmax_recip_trunc': ('rtl', "precip <= dv_qn + 40'(dv_up);", "precip <= dv_qn;"),
        'kv4_zero_extend': ('rtl', "return 8'($signed(w[j*CB +: CB]));", "return 8'(w[j*CB +: CB]);"),
        'block_recip_trunc': ('rtl', "vrecip <= 32'(dv_qn + 40'(dv_up));", "vrecip <= 32'(dv_qn);"),
    }),
    'actq': ('tools/decode/run_act_quant_rtl.py', 'rtl/decode/act_quant.sv', {
        'ties_half_up': ('rtl', "((rem24 == 24'h800000) && q[0])", "(rem24 == 24'h800000)"),
        'recip_truncate': ('rtl', "quot_next + 32'(({rem_next", "quot_next + 32'(1'b0 && ({rem_next"),
        'peak_off_by_one': ('rtl', "if (mag_in > peak) peak <= mag_in;", "if (mag_in > peak) peak <= mag_in + 32'd1;"),
        'qmax7_clamp_8': ('rtl', "(q7 ? 64'd7 : 64'd127)) q = q7 ? 64'd7", "(q7 ? 64'd8 : 64'd127)) q = q7 ? 64'd8"),
        'qmax7_numerator_127': ('rtl', "qmax7_r ? 31'(7) << 24", "qmax7_r ? 31'(127) << 24"),
    }),
    'gemv': ('tools/decode/run_gemv_rtl.py', 'rtl/decode/gemv_unit.sv', {
        'ties_half_up': ('rtl', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'no_div127_up_correction': ('rtl', "if (err2 > 66'sd127)       q5 <= q4 + 64'd1;", "if (err2 > 66'sd127)       q5 <= q4;",
                                    'equivalent: INV127_Q32 = 33818641 > 2^32/127, so the estimate never falls below the true quotient and the +1 branch is unreachable'),
        'no_div127_down_correction': ('rtl', "else if (err2 < -66'sd127) q5 <= q4 - 64'd1;", "else if (err2 < -66'sd127) q5 <= q4;"),
        'drop_last_beat_product': ('rtl', "snap_acc[i]   <= acc[i] + prod[i];", "snap_acc[i]   <= acc[i];"),
        'int4_nibble_swap': ('rtl', "w = nibble_hi ? {{4{b[7]}}, b[7:4]} : {{4{b[3]}}, b[3:0]};", "w = nibble_hi ? {{4{b[3]}}, b[3:0]} : {{4{b[7]}}, b[7:4]};"),
    }),
}


def main():
    path = ROOT / 'bench/results/decode_fault_injection.json'
    results = {}
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
    merged = json.loads(path.read_text()) if path.exists() else {}   # re-read: other runs may have written
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2) + '\n')


if __name__ == '__main__':
    main()
