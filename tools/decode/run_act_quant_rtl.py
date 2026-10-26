"""Differential test: rtl/decode/act_quant.sv vs captured runq_runtime._quantize groups."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import compile_and_run, ROOT, VVP

GOLDEN = ROOT / 'build/decode/golden_act_quant'

TB = r"""`timescale 1ns/1ps
module act_quant_tb;
  localparam int NG = %(ng)d, NE = %(ne)d;
  logic clk = 0, rst = 1, in_valid = 0, in_last = 0;
  logic signed [31:0] raw_in = 0;
  logic in_ready, code_valid, done; logic signed [7:0] code_out; logic [31:0] scale_out;
  logic [31:0] raw_mem [NE]; logic [7:0] code_mem [NE]; logic [31:0] scale_mem [NG]; logic [7:0] len_mem [NG];
  integer base = 0, errors = 0, real_errors = 0, cycles = 0, got, shown = 0;
  act_quant dut(.clk, .rst, .in_valid, .in_last, .raw_in, .in_ready, .code_valid, .code_out, .scale_out, .done);
  always #1 clk = ~clk;
  initial begin
    $readmemh("%(raw)s", raw_mem); $readmemh("%(code)s", code_mem);
    $readmemh("%(scale)s", scale_mem); $readmemh("%(len)s", len_mem);
    @(posedge clk); #0.1 rst = 0;
    for (int g = 0; g < NG; g++) begin
      int n, sent, bad; n = len_mem[g]; sent = 0; got = 0; bad = 0;
      while (!done) begin
        in_valid = sent < n; in_last = sent == n - 1; raw_in = in_valid ? raw_mem[base + sent] : 0;
        @(posedge clk); cycles++;
        if (in_valid && in_ready) sent++;
        if (code_valid) begin
          if (got >= n || code_out !== code_mem[base + got]) begin
            bad++;
            if (shown < 5) begin shown++; $display("MISMATCH group=%%0d i=%%0d got=%%0d exp=%%0d", g, got, code_out, $signed(code_mem[base + got])); end
          end
          got++;
        end
        #0.1;
      end
      in_valid = 0; in_last = 0; @(posedge clk); #0.1;
      if (got != n) bad++;
      if (scale_out !== scale_mem[g]) begin
        bad++; if (shown < 5) begin shown++; $display("SCALE group=%%0d got=%%0d exp=%%0d", g, scale_out, scale_mem[g]); end
      end
      errors += bad; if (g < %(nreal)d) real_errors += bad;
      base += n;
    end
    $display("SUMMARY groups=%%0d elements=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NG, NE, errors, real_errors, cycles);
    $finish;
  end
endmodule
"""


def build(golden, man):
    raw, code, scale, lens = [], [], [], []
    for r in man['records']:
        rd = lambda k: bytes.fromhex(''.join((golden / f"{r['id']}_{k}.hex").read_text().split()))
        raw += np.frombuffer(rd('r'), '<i4').tolist(); code += np.frombuffer(rd('c'), np.int8).tolist()
        scale += np.frombuffer(rd('s'), '<i4').tolist(); lens.append(r['length'])
    base = ROOT / 'build/decode'; files = {}
    for name, vals, w in (('raw', raw, 32), ('code', code, 8), ('scale', scale, 32), ('len', lens, 8)):
        path = base / f'actq_{name}.mem'; path.write_text(''.join(f'{v & ((1 << w) - 1):0{w // 4}x}\n' for v in vals))
        files[name] = path.as_posix()
    tb = base / 'act_quant_tb.sv'
    tb.write_text(TB % dict(ng=len(lens), ne=len(raw), nreal=man['real_groups'], **files))
    return tb, lens


def summary(out):
    line = next((x for x in out.splitlines() if x.startswith('SUMMARY')), '')
    return line, (dict(t.split('=') for t in line.split()[1:]) if line else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', type=Path, default=ROOT / 'rtl/decode/act_quant.sv')
    ap.add_argument('--output', type=Path, default=ROOT / 'bench/results/decode_act_quant_rtl.json')
    a = ap.parse_args()
    if not (GOLDEN / 'manifest.json').exists():
        subprocess.run([sys.executable, str(ROOT / 'tools/decode/golden_act_quant.py')], check=True)
    man_bytes = (GOLDEN / 'manifest.json').read_bytes(); man = json.loads(man_bytes)
    tb, lens = build(GOLDEN, man)
    rc, out = compile_and_run([a.rtl], 'act_quant_tb', tb)
    line, kv = summary(out)
    mrc, mout = compile_and_run([a.rtl], 'act_quant_tb', tb, defines=('ACTQ_MUTATE',))
    mline, mkv = summary(mout)
    expected_mut = sum(1 for n in lens if n > 5)
    errors, merrors = int(kv.get('errors', -1)), int(mkv.get('errors', -1))
    o = {
        'status': 'pass' if rc == 0 and errors == 0 and merrors == expected_mut else 'fail',
        'golden_source': man['source'], 'manifest_sha256': hashlib.sha256(man_bytes).hexdigest(),
        'real_groups': man['real_groups'], 'real_elements': man['real_elements'],
        'random_directed_groups': man['random_directed_groups'], 'random_directed_elements': man['random_directed_elements'],
        'mismatch_count': errors, 'real_mismatch_count': int(kv.get('real_errors', -1)),
        'first_mismatches': [x for x in out.splitlines() if x.startswith(('MISMATCH', 'SCALE'))],
        'mutation_mismatch_count': merrors, 'mutation_expected': expected_mut,
        'cycles_total': int(kv.get('cycles', 0)), 'cycles_per_group': int(kv.get('cycles', 0)) / len(lens),
        'vvp_command': f'{VVP} build/decode/act_quant_tb.vvp',
        'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
        'mutation_stdout_sha256': hashlib.sha256(mout.encode()).hexdigest(),
        'summary': line, 'mutation_summary': mline,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in ('status', 'mismatch_count', 'real_mismatch_count', 'mutation_mismatch_count', 'mutation_expected', 'cycles_per_group')}))


if __name__ == '__main__':
    main()
