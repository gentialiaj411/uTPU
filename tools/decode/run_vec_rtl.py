"""Differential test: rtl/decode/vec_alu.sv vs captured RoPE / SwiGLU / add / embedding-scale cases."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from harness import ROOT, load_cases, write_mem, run_differential, finish, ARTIFACT_SOURCES

TB = r"""`timescale 1ns/1ps
module %(unit)s_tb;
  localparam int NC = %(nc)d, NB = %(nb)d;
  logic clk = 0, rst = 1, in_valid = 0, op_start = 0;
  logic [1:0] op; logic [15:0] rope_base; logic [5:0] rope_half;
  logic [31:0] silu_mult, swiglu_mult, m; logic [6:0] silu_shift, swiglu_shift, sh;
  logic signed [31:0] a, b; logic in_ready, out_valid; logic signed [31:0] out0, out1;
  logic [103:0] beat_mem [NB]; logic [63:0] exp_mem [NB]; logic [127:0] cfg_mem [NC];
  integer base = 0, errors = 0, real_errors = 0, cycles = 0, shown = 0;
  vec_alu #(.ROPE_DEPTH(%(depth)d), .ROPE_CS("%(rope)s")) dut(.clk, .rst, .op, .op_start, .rope_base, .rope_half, .silu_mult, .silu_shift, .swiglu_mult,
              .swiglu_shift, .in_valid, .a, .b, .m, .sh, .in_ready, .out_valid, .out0, .out1);
  always #1 clk = ~clk;
  initial begin
    $readmemh("%(beats)s", beat_mem); $readmemh("%(exp)s", exp_mem); $readmemh("%(cfg)s", cfg_mem);
    {a, b, m, sh} = '0;
    @(posedge clk); #0.1 rst = 0;
    for (int c = 0; c < NC; c++) begin
      int n, sent, got, bad;
      {n, op, rope_base, rope_half, silu_mult, silu_shift, swiglu_mult, swiglu_shift} = cfg_mem[c][117:0];
      op_start = 1; @(posedge clk); #0.1 op_start = 0;
      sent = 0; got = 0; bad = 0;
      while (got < n) begin
        in_valid = sent < n;
        {a, b, m, sh} = in_valid ? beat_mem[base + sent] : 104'd0;
        @(posedge clk); cycles++;
        if (in_valid && in_ready) sent++;
        if (out_valid) begin
          if (out0 !== exp_mem[base + got][63:32] || (op == 0 && out1 !== exp_mem[base + got][31:0])) begin
            bad++;
            if (shown < 8) begin shown++; $display("MISMATCH case=%%0d op=%%0d i=%%0d got=%%0d,%%0d exp=%%0d,%%0d", c, op, got,
              out0, out1, $signed(exp_mem[base + got][63:32]), $signed(exp_mem[base + got][31:0])); end
          end
          got++;
        end
        #0.1;
        if (cycles > 50000000) begin errors++; $display("TIMEOUT case=%%0d: outputs never completed", c);
          $display("SUMMARY cases=%%0d beats=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NC, NB, errors, real_errors, cycles); $finish; end
      end
      in_valid = 0;
      errors += bad; if (c < %(nreal)d) real_errors += bad;
      base += n;
    end
    $display("SUMMARY cases=%%0d beats=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NC, NB, errors, real_errors, cycles);
    $finish;
  end
endmodule
"""


GEOMETRIES = (   # (case set, unit name, RoPE ROM, ROM depth = positions * half-width)
    ('vec', 'vec', 'rtl/decode/luts/rope_cs_q15_h8.mem', 512 * 4),               # stories260K: HS=8, half 4
    ('vec15', 'vec15', 'rtl/decode/luts/rope_cs_q15_h48_p128.mem', 128 * 24),    # stories15M: HS=48, half 24, MAX_SEQ 128
)


def run_geometry(a, case_set, unit, rope, depth):
    data, sha = load_cases(case_set)
    cases, meta = data['cases'], data['meta']
    beats, exp, cfg = [], [], []
    for c in cases:
        for (x, y, m, s), (o0, o1) in zip(c['beats'], c['out']):
            beats.append(((x & 0xFFFFFFFF) << 71) | ((y & 0xFFFFFFFF) << 39) | ((m & 0xFFFFFFFF) << 7) | s)
            exp.append(((o0 & 0xFFFFFFFF) << 32) | (o1 & 0xFFFFFFFF))
        k = c['cfg']; word = len(c['beats'])
        for v, w in ((c['op'], 2), (k['rope_base'], 16), (k['rope_half'], 6), (k['silu_mult'], 32), (k['silu_shift'], 7),
                     (k['swiglu_mult'], 32), (k['swiglu_shift'], 7)):
            word = (word << w) | v
        cfg.append(word)
    files = dict(beats=write_mem(f'{unit}_beats', beats, 104), exp=write_mem(f'{unit}_exp', exp, 64), cfg=write_mem(f'{unit}_cfg', cfg, 128))
    tb = TB % dict(nc=len(cases), nb=len(beats), nreal=meta['real_cases'], rope=rope, depth=depth, unit=unit, **files)
    return run_differential(unit, [a.pkg, a.rtl], tb, 'VEC_MUTATE', len(beats),
                            {'golden_source': meta['source'], 'geometry': meta['geometry'], 'rope_rom': rope, 'rope_half': meta['rope_half'],
                             'rope_positions_real': meta['rope_positions_real'], 'real_cases': meta['real_cases'],
                             'directed_random_cases': meta['directed_random_cases'], 'beats': len(beats),
                             'ops': {n: sum(1 for c in cases if c['op'] == i) for i, n in enumerate(('rope', 'swiglu', 'add', 'mulshift'))},
                             'kinds': sorted({c['kind'] for c in cases})}, sha, str(Path(a.output).with_name(f'{Path(a.output).stem}_{unit}.json')), write=False,
                            overrides={'rtl/decode/vec_alu.sv': a.rtl, 'rtl/decode/decode_pkg.sv': a.pkg},
                            sources=ARTIFACT_SOURCES['decode_vec_alu_rtl'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', default=str(ROOT / 'rtl/decode/vec_alu.sv'))
    ap.add_argument('--pkg', default=str(ROOT / 'rtl/decode/decode_pkg.sv'))
    ap.add_argument('--output', default=str(ROOT / 'bench/results/decode_vec_alu_rtl.json'))
    ap.add_argument('--sim', choices=('icarus','verilator'), default=__import__('sim').SIMULATOR, help='simulation backend (default icarus)')
    a = ap.parse_args()
    runs = {unit: run_geometry(a, cs, unit, rope, depth) for cs, unit, rope, depth in GEOMETRIES}
    o = {'status': 'pass' if all(r['status'] == 'pass' for r in runs.values()) else 'fail',
         'mismatch_count': sum(r['mismatch_count'] for r in runs.values()),
         'real_mismatch_count': sum(r['real_mismatch_count'] for r in runs.values()),
         'mutation_mismatch_count': sum(r['mutation_mismatch_count'] for r in runs.values()),
         'mutation_expected': sum(r['mutation_expected'] for r in runs.values()),
         'cycles_total': sum(r['cycles_total'] for r in runs.values()),
         'first_mismatches': [m for r in runs.values() for m in r['first_mismatches']][:8],
         'beats': sum(r['beats'] for r in runs.values()),
         'geometries': runs,
         'provenance': {k: v for r in runs.values() for k, v in r['provenance'].items()}}
    finish(o, a.output, ('status', 'mismatch_count', 'real_mismatch_count', 'mutation_mismatch_count', 'mutation_expected', 'cycles_total', 'first_mismatches'))


if __name__ == '__main__':
    main()
