"""15M-shape differential test: attention.sv at HS=48, 6 layers, 6 KV heads, MAX_SEQ 128, KV4 (CB=4, 224-bit words)
vs stories15M runtime decode attention up to position 127 (golden_attn15.py) plus random 15M-shape caches.
"""
from __future__ import annotations
import argparse, hashlib, json
from harness import ROOT, BUILD, load_cases, write_mem, parse_summary
from sim import compile_and_run, VVP

MAX_SEQ = 128
LAYERS, KVH, LW, PW, HW = 6, 6, 3, 7, 3

TB = r"""`timescale 1ns/1ps
module attn_tb;
  localparam int NG = %(ng)d, NK = %(nk)d, NC = %(nc)d, HS = %(hs)d;
  logic clk = 0, rst = 1, kv_we = 0, kv_is_v = 0, start = 0;
  logic [2:0] kv_layer, layer; logic [6:0] kv_pos, pos; logic [2:0] kv_head, head;
  logic [HS*8-1:0] kv_codes, q_codes; logic [31:0] kv_scale, q_scale;
  logic busy, out_valid, done; logic signed [31:0] out;
  logic [511:0] kv_mem [NK]; logic [511:0] call_mem [NC]; logic [31:0] exp_mem [NC*HS]; logic [63:0] grp_mem [NG];
  integer ki = 0, ci = 0, errors = 0, real_errors = 0, cycles = 0, shown = 0;
  attention #(.LAYERS(6), .MAX_SEQ(%(maxseq)d), .KV_HEADS(6), .HS(HS), .CB(%(cb)d), .LANES(8)) dut (
    .clk, .rst, .score_mult(32'd%(smult)d), .score_shift(7'd%(sshift)d), .kv4(1'b%(kv4)d), .kv_we, .kv_is_v,
    .kv_layer, .kv_pos, .kv_head, .kv_codes, .kv_scale, .start, .layer, .head, .pos, .q_codes, .q_scale,
    .busy, .out_valid, .out, .done);
  always #1 clk = ~clk;
  initial begin
    $readmemh("%(kv)s", kv_mem); $readmemh("%(calls)s", call_mem); $readmemh("%(exp)s", exp_mem); $readmemh("%(grp)s", grp_mem);
    @(posedge clk); #0.1 rst = 0;
    for (int g = 0; g < NG; g++) begin
      logic isreal; logic [23:0] nk, ncalls;
      {isreal, nk, ncalls} = grp_mem[g][48:0];
      for (int e = 0; e < nk; e++) begin
        {kv_is_v, kv_layer, kv_pos, kv_head, kv_scale, kv_codes} = kv_mem[ki + e][HS*8+45:0];
        kv_we = 1; @(posedge clk); #0.1 kv_we = 0;
      end
      ki += nk;
      for (int c = 0; c < ncalls; c++) begin
        int got, bad;
        {layer, pos, head, q_scale, q_codes} = call_mem[ci][HS*8+44:0];
        start = 1; @(posedge clk); #0.1 start = 0; got = 0; bad = 0;
`ifdef ATTN_DEBUG
        if (ci == 0) begin
          wait (dut.state == dut.SM_RD); for (int t = 0; t <= pos; t++) $display("DBG score[%%0d]=%%0d", t, dut.sbuf[t]);
          wait (dut.state == dut.AV_VRD); $display("DBG precip=%%0d pbuf0=%%0d", dut.precip, dut.pbuf[0]);
          wait (dut.state == dut.AV_RDIV); $display("DBG peak=%%0d vbuf0=%%0d vbuf1=%%0d", dut.peak, dut.vbuf[0], dut.vbuf[1]);
          wait (dut.state == dut.AV_ARD); $display("DBG vrecip=%%0d", dut.vrecip);
          wait (dut.state == dut.AV_FMUL); $display("DBG abuf0=%%0d abuf1=%%0d", dut.abuf[0], dut.abuf[1]);
          wait (dut.state == dut.AV_FDWAIT); #3 $display("DBG mp_p=%%0d", dut.mp_p);
          wait (dut.state == dut.OUT); $display("DBG raw0=%%0d raw1=%%0d", dut.raw[0], dut.raw[1]);
        end
`endif
        for (int guard = 0; got < HS && guard < 2000000; guard++) begin
          @(posedge clk); cycles++;
          if (out_valid) begin
            if (out !== exp_mem[ci*HS + got]) begin
              bad++;
              if (shown < 8) begin shown++; $display("MISMATCH call=%%0d j=%%0d got=%%0d exp=%%0d", ci, got, out, $signed(exp_mem[ci*HS + got])); end
            end
            got++;
          end
          #0.1;
        end
        if (got != HS) bad += HS;
        @(posedge clk); #0.1;
        errors += bad; if (isreal) real_errors += bad;
        ci++;
      end
    end
    $display("SUMMARY calls=%%0d errors=%%0d real_errors=%%0d cycles=%%0d", NC, errors, real_errors, cycles);
    $finish;
  end
endmodule
"""


def build(groups, meta, kv_bits, tag):
    hs = meta['head_size']
    kv, calls, exp, grp = [], [], [], []
    for g in groups:
        if g['kv_bits'] != kv_bits:
            continue
        d = g['kv']
        n0 = len(kv)
        # Write in decode order (position-major, then layer, then K before V): the order the engine sees in
        # a real run. Address order would mask bank-aliasing writes, which only land on later addresses.
        for p in range(MAX_SEQ):
            for layer in range(meta['n_layers']):
                for is_v, codes, scales in ((0, d['key_code'], d['key_scale']), (1, d['value_code'], d['value_scale'])):
                    if p >= len(codes[layer]):
                        continue
                    for h in range(meta['n_kv_heads']):
                        word = 0
                        for jj, cval in enumerate(codes[layer][p][h * hs:(h + 1) * hs]):
                            word |= (cval & 0xFF) << (8 * jj)
                        head = (((is_v << 3 | layer) << 7 | p) << 3 | h) << 32 | (scales[layer][p][h] & 0xFFFFFFFF)
                        kv.append(head << (8 * hs) | word)
        for c in g['calls']:
            word = 0
            for jj, cval in enumerate(c['q_code']):
                word |= (cval & 0xFF) << (8 * jj)
            calls.append((((c['layer'] << 7 | c['pos']) << 3 | c['kv_head']) << 32 | c['q_scale']) << (8 * hs) | word)
            exp += c['out']
        grp.append((int(g['calls'][0]['kind'].startswith('real')) << 48) | (len(kv) - n0) << 24 | len(g['calls']))
    files = dict(kv=write_mem(f'attn15_kv_{tag}', kv, 512), calls=write_mem(f'attn15_calls_{tag}', calls, 512),
                 exp=write_mem(f'attn15_exp_{tag}', exp, 32), grp=write_mem(f'attn15_grp_{tag}', grp, 64))
    return TB % dict(ng=len(grp), nk=len(kv), nc=len(calls), hs=hs, maxseq=MAX_SEQ, cb=8 if kv_bits == 8 else 4,
                     smult=meta['score_multiplier'], sshift=meta['score_shift'], kv4=int(kv_bits == 4), **files), len(calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', default=str(ROOT / 'rtl/decode/attention.sv'))
    ap.add_argument('--pkg', default=str(ROOT / 'rtl/decode/decode_pkg.sv'))
    ap.add_argument('--output', default=str(ROOT / 'bench/results/decode_attention15_rtl.json'))
    a = ap.parse_args()
    data, sha = load_cases('attn15')
    groups, meta = data['cases'], data['meta']
    srcs = [a.pkg, ROOT / 'rtl/decode/mul_pipe.sv', ROOT / 'rtl/decode/div_const_pipe.sv', a.rtl]
    res = {'errors': 0, 'real_errors': 0, 'mut_errors': 0, 'cycles': 0, 'calls': 0, 'first': [], 'stdout_sha': [], 'summaries': []}
    for kv_bits in (4,):
        text, ncalls = build(groups, meta, kv_bits, f'kv{kv_bits}')
        tb = BUILD / f'attn15_kv{kv_bits}_tb.sv'
        tb.write_text(text)
        rc, out = compile_and_run(srcs, 'attn_tb', tb)
        _, kv = parse_summary(out)
        _, mout = compile_and_run(srcs, 'attn_tb', tb, defines=('ATTN_MUTATE',))
        _, mkv = parse_summary(mout)
        res['errors'] += int(kv.get('errors', 10**9)) if rc == 0 else 10**9
        res['real_errors'] += int(kv.get('real_errors', 10**9))
        res['mut_errors'] += int(mkv.get('errors', -1))
        res['cycles'] += int(kv.get('cycles', 0)); res['calls'] += ncalls
        res['first'] += [x for x in out.splitlines() if x.startswith('MISMATCH')][:4]
        res['stdout_sha'].append(hashlib.sha256(out.encode()).hexdigest())
        res['summaries'].append(f'kv{kv_bits}: ' + next((x for x in out.splitlines() if x.startswith('SUMMARY')), out[-300:]))
    expected_mut = res['calls'] * meta['head_size']
    o = {'status': 'pass' if res['errors'] == 0 and res['mut_errors'] == expected_mut else 'fail',
         'golden_sha256': sha, 'golden_source': meta['source'], 'real_head_calls': meta['real_cases'],
         'random_head_calls': meta['directed_random_cases'], 'output_elements': expected_mut,
         'mismatch_count': res['errors'], 'real_mismatch_count': res['real_errors'], 'first_mismatches': res['first'],
         'mutation_mismatch_count': res['mut_errors'], 'mutation_expected': expected_mut,
         'cycles_total': res['cycles'], 'cycles_per_head_call': res['cycles'] / max(1, res['calls']),
         'vvp_command': f'{VVP} build/decode/attn_tb.vvp (15M shape: HS48 L6 KVH6 SEQ128 CB=4)',
         'sim_stdout_sha256': res['stdout_sha'], 'summary': res['summaries']}
    ROOT.joinpath(a.output).parent.mkdir(parents=True, exist_ok=True)
    open(a.output, 'w').write(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in ('status', 'mismatch_count', 'real_mismatch_count', 'mutation_mismatch_count', 'mutation_expected', 'cycles_per_head_call', 'first_mismatches', 'summary')}))


if __name__ == '__main__':
    main()
