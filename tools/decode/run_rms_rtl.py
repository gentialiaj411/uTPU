"""Differential test: rtl/decode/rmsnorm.sv vs captured runq_runtime._rms_high cases."""
from __future__ import annotations
import argparse, json
from harness import ROOT, load_cases, write_mem, run_differential

TB = r"""`timescale 1ns/1ps
module rms_tb;
  localparam int NC = %(nc)d, NE = %(ne)d;
  logic clk = 0, rst = 1, in_valid = 0, in_last = 0;
  logic [32:0] recip_d; logic [4:0] gfb; logic [31:0] mult; logic signed [63:0] eps; logic [6:0] shift;
  logic signed [31:0] x_in = 0; logic [15:0] gamma_in = 0;
  logic in_ready, out_valid, done; logic signed [31:0] out;
  logic [47:0] in_mem [NE]; logic [31:0] exp_mem [NE]; logic [159:0] cfg_mem [NC];
  integer base = 0, errors = 0, real_errors = 0, cycles = 0, shown = 0;
  rmsnorm dut(.clk, .rst, .recip_d, .gfb, .mult, .shift, .eps, .in_valid, .in_last, .x_in, .gamma_in,
              .in_ready, .out_valid, .out, .done);
  always #1 clk = ~clk;
  initial begin
    $readmemh("%(inp)s", in_mem); $readmemh("%(exp)s", exp_mem); $readmemh("%(cfg)s", cfg_mem);
    @(posedge clk); #0.1 rst = 0;
    for (int c = 0; c < NC; c++) begin
      int n, sent, got, bad;
      {recip_d, gfb, mult, shift, eps} = cfg_mem[c][140:0]; n = cfg_mem[c][156:141];
      sent = 0; got = 0; bad = 0;
      while (!done) begin
        in_valid = sent < n; in_last = sent == n - 1;
        {x_in, gamma_in} = in_valid ? in_mem[base + sent] : 48'd0;
        @(posedge clk); cycles++;
        if (in_valid && in_ready) sent++;
        if (out_valid) begin
          if (got >= n || out !== exp_mem[base + got]) begin
            bad++;
            if (shown < 8) begin shown++; $display("MISMATCH case=%%0d i=%%0d got=%%0d exp=%%0d", c, got, out, $signed(exp_mem[base + got])); end
          end
          got++;
        end
        #0.1;
      end
      in_valid = 0; in_last = 0; @(posedge clk); #0.1;
      if (got != n) bad += n;
      errors += bad; if (c < %(nreal)d) real_errors += bad;
      base += n;
    end
    $display("SUMMARY cases=%%0d elements=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NC, NE, errors, real_errors, cycles);
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', default=str(ROOT / 'rtl/decode/rmsnorm.sv'))
    ap.add_argument('--pkg', default=str(ROOT / 'rtl/decode/decode_pkg.sv'))
    ap.add_argument('--output', default=str(ROOT / 'bench/results/decode_rms_rtl.json'))
    a = ap.parse_args()
    data, sha = load_cases('rms')
    cases, meta = data['cases'], data['meta']
    inp, exp, cfg = [], [], []
    for c in cases:
        inp += [((x & 0xFFFFFFFF) << 16) | g for x, g in zip(c['x'], c['gamma'])]
        exp += c['out']
        word = c['recip_d']
        for v, w in ((c['gfb'], 5), (c['mult'], 32), (c['shift'], 7), (c['eps'] & ((1 << 64) - 1), 64)):
            word = (word << w) | v
        cfg.append((len(c['x']) << 141) | word)
    files = dict(inp=write_mem('rms_in', inp, 48), exp=write_mem('rms_exp', exp, 32), cfg=write_mem('rms_cfg', cfg, 160))
    tb = TB % dict(nc=len(cases), ne=len(exp), nreal=meta['real_cases'], **files)
    o = run_differential('rms', [a.pkg, ROOT / 'rtl/decode/seq_mul.sv', a.rtl], tb, 'RMS_MUTATE', len(exp),
                         {'golden_source': meta['source'], 'real_cases': meta['real_cases'],
                          'directed_random_cases': meta['directed_random_cases'], 'elements': len(exp),
                          'kinds': sorted({c['kind'] for c in cases})}, sha, a.output)
    print(json.dumps({k: o[k] for k in ('status', 'mismatch_count', 'real_mismatch_count', 'mutation_mismatch_count', 'mutation_expected', 'first_mismatches')}))


if __name__ == '__main__':
    main()
