"""Differential GEMV test: rtl/decode/gemv_unit.sv vs captured runq_runtime._linear_high tiles."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import compile_and_run, ROOT, VVP

LANES = 16


def load(golden, rec, kind, dtype):
    return np.frombuffer(bytes.fromhex((golden / f"{rec['id']}_{kind}.hex").read_text().strip()), dtype=dtype)


def tile_streams(golden, rec):
    """Return (beats, chunks, expected) for one tile in the RTL's input format."""
    k, rows, qmax = rec['k'], rec['rows'], rec['qmax']
    act = load(golden, rec, 'a', np.int8).reshape(-1)
    act_scales = load(golden, rec, 'as', '<i4').reshape(-1)
    wbytes = load(golden, rec, 'w', np.uint8)
    mult = load(golden, rec, 'wm', '<i8').reshape(rows, -1)
    shift = load(golden, rec, 'ws', '<i8').reshape(rows, -1)
    expected = load(golden, rec, 'o', '<i4')
    begins, ends = rec['groups']
    int4 = qmax == 7
    stride = (k + 1) // 2 if int4 else k
    chunk_ends = {}
    for g, (b, e) in enumerate(zip(begins, ends)):
        cb = b
        while cb < e:
            ce = min(e, (cb // 64 + 1) * 64)
            chunk_ends[ce - 1] = (g, cb // 64)
            cb = ce
    beats, chunks = [], []
    for pos in range(k):
        word = 0
        for lane in range(LANES):
            byte = 0
            if lane < rows:
                byte = int(wbytes[lane * stride + (pos // 2 if int4 else pos)])
            word |= byte << (8 * lane)
        last_chunk = pos in chunk_ends
        flags = (int(pos == k - 1) << 3) | (int(last_chunk) << 2) | (int(int4 and pos % 2 == 1) << 1) | int(int4)
        beats.append((flags << 136) | ((int(act[pos]) & 0xFF) << 128) | word)
        if last_chunk:
            g, ag = chunk_ends[pos]
            m = sh = 0
            for lane in range(rows):
                m |= (int(mult[lane, g]) & 0xFFFFFFFF) << (32 * lane)
                sh |= (int(shift[lane, g]) & 0xFF) << (8 * lane)
            chunks.append((int(act_scales[ag]) & 0xFFFFFFFF) << 640 | sh << 512 | m)
    exp = 0
    for lane in range(rows):
        exp |= (int(expected[lane]) & 0xFFFFFFFF) << (32 * lane)
    return beats, chunks, exp, rows


TB = r"""
module gemv_tb;
  localparam int NT = %(nt)d, NB = %(nb)d, NC = %(nc)d;
  logic clk = 0, rst = 1, start = 0, in_valid = 0;
  logic [139:0] beat_mem [NB];
  logic [671:0] chunk_mem [NC];
  logic [511:0] exp_mem [NT];
  logic [31:0] meta_mem [NT];          // {rows[7:0], beats[23:0]}
  logic [139:0] b; logic [671:0] c;
  logic in_ready, busy, done; logic [511:0] result;
  integer bi = 0, ci = 0, errors = 0, real_errors = 0, cycles = 0, mism = 0;
  decode_gemv_unit dut(.clk, .rst, .start, .in_valid, .in_ready,
    .int4_mode(b[136]), .nibble_hi(b[137]), .act_code(b[135:128]), .weight_bytes(b[127:0]),
    .chunk_last(b[138]), .tile_last(b[139]), .act_scale(c[671:640]), .multipliers(c[511:0]),
    .shifts(c[639:512]), .busy, .done, .result);
  always #1 clk = ~clk;
  initial begin
    $readmemh("%(beats)s", beat_mem); $readmemh("%(chunks)s", chunk_mem);
    $readmemh("%(exp)s", exp_mem); $readmemh("%(meta)s", meta_mem);
    b = '0; c = '0;
    @(posedge clk); #0.1 rst = 0;
    for (int t = 0; t < NT; t++) begin
      int nbeats, rows, sent;
      nbeats = meta_mem[t][23:0]; rows = meta_mem[t][31:24]; sent = 0;
      start = 1; @(posedge clk); #0.1 start = 0;
      while (!done) begin
        if (sent < nbeats) begin b = beat_mem[bi]; c = chunk_mem[ci]; in_valid = 1; end
        else in_valid = 0;
        @(posedge clk); cycles++;
        if (in_valid && in_ready) begin
          if (b[138]) ci++;
          bi++; sent++;
        end
        #0.1;
      end
      in_valid = 0;
      for (int r = 0; r < rows; r++)
        if (result[r*32 +: 32] !== exp_mem[t][r*32 +: 32]) begin
          errors++; if (t < %(nreal)d) real_errors++;
          if (mism < 5) $display("MISMATCH tile=%%0d row=%%0d got=%%0d exp=%%0d", t, r,
            $signed(result[r*32 +: 32]), $signed(exp_mem[t][r*32 +: 32]));
          mism++;
        end
    end
    $display("SUMMARY tiles=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NT, errors, real_errors, cycles);
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--golden', type=Path, default=ROOT / 'build/decode/golden_gemv')
    ap.add_argument('--output', type=Path, default=ROOT / 'bench/results/decode_gemv_rtl.json')
    ap.add_argument('--rtl', type=Path, default=ROOT / 'rtl/decode/gemv_unit.sv')
    a = ap.parse_args()
    man_bytes = (a.golden / 'manifest.json').read_bytes(); man = json.loads(man_bytes)
    recs = man['records'] + man['random_records']
    beats, chunks, exps, meta, elems = [], [], [], [], [0, 0]
    for i, rec in enumerate(recs):
        bt, ch, ex, rows = tile_streams(a.golden, rec)
        beats += bt; chunks += ch; exps.append(ex); meta.append((rows << 24) | len(bt))
        elems[i >= len(man['records'])] += rows
    base = ROOT / 'build/decode'; base.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, vals, width in (('beats', beats, 140), ('chunks', chunks, 672), ('exp', exps, 512), ('meta', meta, 32)):
        path = base / f'gemv_{name}.mem'
        path.write_text(''.join(f'{v:0{(width + 3) // 4}x}\n' for v in vals))
        files[name] = path.as_posix()
    tb = base / 'gemv_tb.sv'
    tb.write_text(TB % dict(nt=len(recs), nb=len(beats), nc=len(chunks), nreal=len(man['records']), **files))
    src = [a.rtl]

    def parse(out):
        line = next((x for x in out.splitlines() if x.startswith('SUMMARY')), '')
        kv = dict(t.split('=') for t in line.split()[1:]) if line else {}
        return line, kv

    rc, out = compile_and_run(src, 'gemv_tb', tb)
    summary, kv = parse(out)
    mrc, mout = compile_and_run(src, 'gemv_tb', tb, defines=('GEMV_MUTATE',))
    msummary, mkv = parse(mout)
    tiles_with_row3 = sum(1 for r in recs if r['rows'] > 3)
    errors = int(kv.get('errors', -1)); merrors = int(mkv.get('errors', -1))
    o = {
        'status': 'pass' if rc == 0 and errors == 0 and merrors == tiles_with_row3 else 'fail',
        'golden_source': man['source'],
        'manifest_sha256': hashlib.sha256(man_bytes).hexdigest(),
        'real_tiles': len(man['records']), 'random_tiles': len(man['random_records']),
        'real_output_elements': elems[0], 'random_output_elements': elems[1],
        'mismatch_count': errors, 'real_mismatch_count': int(kv.get('real_errors', -1)),
        'first_mismatches': [x for x in out.splitlines() if x.startswith('MISMATCH')],
        'mutation_mismatch_count': merrors, 'mutation_expected': tiles_with_row3,
        'beats': len(beats), 'chunks': len(chunks),
        'cycles_total': int(kv.get('cycles', 0)), 'cycles_per_tile': int(kv.get('cycles', 0)) / len(recs),
        'observed_max_widths': man.get('observed_max_widths'),
        'vvp_command': f'{VVP} build/decode/gemv_tb.vvp',
        'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
        'mutation_stdout_sha256': hashlib.sha256(mout.encode()).hexdigest(),
        'summary': summary, 'mutation_summary': msummary,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in ('status', 'mismatch_count', 'mutation_mismatch_count', 'mutation_expected', 'cycles_per_tile')}))


if __name__ == '__main__':
    main()
