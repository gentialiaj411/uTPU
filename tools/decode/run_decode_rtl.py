"""End-to-end decode: rtl/decode/top_decode.sv vs RunQIntegerLlama.forward on stories260K.

Golden: greedy decode with the runtime's own forward() (full logits) for each prompt; the RTL gets
the same token at each position (prompt tokens, then the golden's argmax) and must reproduce every
logit and the argmax. The testbench flash model serves the packed image (pack_decode_image.py) one
byte per `period` cycles and drops st_valid during the seek cycle.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from harness import ROOT, BUILD, load_runtime_260k, write_mem, parse_summary
from sim import compile_and_run, VVP
from pack_decode_image import pack

SRCS = ['decode_pkg.sv', 'seq_mul.sv', 'mul_pipe.sv', 'div_const_pipe.sv', 'gemv_unit.sv', 'act_quant.sv',
        'rmsnorm.sv', 'vec_alu.sv', 'attention.sv', 'top_decode.sv']

TB = r"""`timescale 1ns/1ps
module e2e_tb;
  localparam int NS = %(ns)d, V = %(vocab)d, NB = %(nbytes)d, PERIOD = %(period)d, LIMIT = %(limit)d;
  logic clk = 0, rst = 1, start = 0, st_valid, st_ready, seek_req, busy, done, logit_valid;
  logic [15:0] token, token_out, logit_idx; logic [%(pw)d-1:0] pos; logic [31:0] seek_addr, ptr = 0;
  logic [7:0] img [NB]; logic signed [31:0] logit_val; logic [47:0] step_mem [NS]; logic [31:0] exp_logits [NS*V];
  integer errors = 0, tok_errors = 0, cycles = 0, shown = 0, nlog, tcyc, phase = 0;
  top_decode #(%(params)s) dut (.clk, .rst, .start, .token, .pos, .seek_req, .seek_addr, .st_valid,
    .st_data(img[ptr]), .st_ready, .busy, .done, .token_out, .logit_valid, .logit_idx, .logit_val);
  integer prof [32]; initial for (int q = 0; q < 32; q++) prof[q] = 0;
  always @(posedge clk) if (dut.busy) prof[dut.step] <= prof[dut.step] + 1;
  always #1 clk = ~clk;
  assign st_valid = (phase == 0) && !seek_req;
  always @(posedge clk) begin
    phase <= (phase + 1) %% PERIOD;
    if (seek_req) ptr <= seek_addr; else if (st_valid && st_ready) ptr <= ptr + 1;
  end
  initial begin
    $readmemh("%(img)s", img); $readmemh("%(steps)s", step_mem); $readmemh("%(logits)s", exp_logits);
    @(posedge clk); #0.1 rst = 0;
    for (int s = 0; s < NS; s++) begin
      token = step_mem[s][47:32]; pos = step_mem[s][%(pw)d-1:0];
      start = 1; @(posedge clk); #0.1 start = 0; nlog = 0; tcyc = 0;
      while (!done && tcyc < LIMIT) begin
        @(posedge clk); cycles++; tcyc++;
        if (logit_valid) begin
          if (logit_val !== exp_logits[s*V + logit_idx]) begin
            errors++;
            if (shown < 8) begin shown++; $display("MISMATCH step=%%0d idx=%%0d got=%%0d exp=%%0d", s, logit_idx, logit_val, $signed(exp_logits[s*V + logit_idx])); end
          end
          nlog++;
        end
        #0.1;
      end
      if (!done) begin errors += V; $display("TIMEOUT step=%%0d after %%0d cycles", s, tcyc); $display("SUMMARY steps=%%0d errors=%%0d token_errors=%%0d cycles=%%0d", NS, errors, tok_errors, cycles); $finish; end
      if (nlog != V) errors += V;
      if (token_out !== step_mem[s][31:16]) begin tok_errors++; if (shown < 8) begin shown++; $display("TOKEN step=%%0d got=%%0d exp=%%0d", s, token_out, step_mem[s][31:16]); end end
      $display("STEP %%0d token=%%0d pos=%%0d out=%%0d cycles=%%0d", s, token, pos, token_out, tcyc);
      @(posedge clk); #0.1;
    end
    for (int q = 0; q < 32; q++) if (prof[q] != 0) $display("PROF %%0d %%0d", q, prof[q]);
    $display("SUMMARY steps=%%0d errors=%%0d token_errors=%%0d cycles=%%0d", NS, errors, tok_errors, cycles);
    $finish;
  end
endmodule
"""


def load_model(model, kv_bits):
    if model == '260k':
        return load_runtime_260k(kv_bits)
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
    pay = np.load(ROOT / 'models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz', allow_pickle=True)['payload'][0]
    pay.pop('dynamic_base_scales')
    return pay, RunQIntegerLlama(pay, ROOT / 'firmware/host/numerics_v2/luts', kv_bits=kv_bits)


def golden(model, kv_bits, prompts, gen):
    exp, rt = load_model(model, kv_bits)
    seqs = []
    for prompt in prompts:
        rt.reset()
        tokens, outs, logits = [], [], []
        tok = prompt[0]
        for p in range(len(prompt) + gen - 1):
            lg = rt.forward(tok, p)
            nxt = int(np.argmax(lg))            # first maximum == runtime's lexsort top-1
            tokens.append(tok); outs.append(nxt); logits.append(lg.astype(np.int64))
            tok = prompt[p + 1] if p + 1 < len(prompt) else nxt
        seqs.append((tokens, outs, logits))
    return exp, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='260k', choices=('260k', '15m'))
    ap.add_argument('--kv-bits', type=int, default=8, choices=(4, 8))
    ap.add_argument('--period', type=int, default=1, help='flash delivers one byte every PERIOD cycles')
    ap.add_argument('--prompts', type=int, default=1)
    ap.add_argument('--gen', type=int, default=8, help='generated tokens per prompt')
    ap.add_argument('--output', default=None)
    ap.add_argument('--top-rtl', default=None, help='replacement top_decode.sv (fault injection)')
    ap.add_argument('--define', action='append', default=[], help='iverilog define (mutation checks)')
    a = ap.parse_args()
    prompt_bank = [[1, 403, 260], [1, 87], [1, 33, 499, 12], [1, 300], [1, 150, 77]] if a.model == '260k' else \
        [[1, 9038, 2501, 263], [1, 3118]]
    prompts = prompt_bank[:a.prompts]
    t0 = time.time()
    exp, seqs = golden(a.model, a.kv_bits, prompts, a.gen)
    img, params = pack(exp)
    steps, logits = [], []
    for tokens, outs, lg in seqs:
        for p, (t, o, l) in enumerate(zip(tokens, outs, lg)):
            steps.append((t << 32) | (o << 16) | p)
            logits += [int(x) for x in l]
    tag = ('' if a.model == '260k' else '15m_') + f'kv{a.kv_bits}_p{a.period}' + ''.join('_' + d for d in a.define) + ('_' + Path(a.top_rtl).stem if a.top_rtl else '')
    files = dict(img=write_mem(f'e2e_img_{a.model}', list(img), 8), steps=write_mem(f'e2e_steps_{tag}', steps, 48),
                 logits=write_mem(f'e2e_logits_{tag}', logits, 32))
    keys = ['D', 'H', 'NL', 'NH', 'NKV', 'HS', 'VOCAB', 'CLS4', 'WQ4', 'WK4', 'WV4', 'WO4', 'W14', 'W34', 'W24',
            'CLS_BASE', 'EMB_SCALE_BASE', 'LAYERS_BASE', 'IMAGE_BYTES', 'CLS_TILE_BYTES', 'CLS_KB', 'CLS_G', 'RECIP_D', 'NORM_MULT',
            'EPS', 'GFB', 'NORM_SHIFT', 'SILU_MULT', 'SWIGLU_MULT', 'SCORE_MULT', 'SILU_SHIFT', 'SWIGLU_SHIFT', 'SCORE_SHIFT']
    max_seq = 64 if a.model == '260k' else 128
    plist = ', '.join(f'.{k}({params[k]})' for k in keys) + f', .MAX_SEQ({max_seq}), .KV4({int(a.kv_bits == 4)})'
    if a.model == '15m':
        plist += ', .ROPE_CS("rtl/decode/luts/rope_cs_q15_h48_p128.mem"), .ROPE_DEPTH(3072)'
    tb = BUILD / f'e2e_{tag}_tb.sv'
    tb.write_text(TB % dict(ns=len(steps), vocab=params['VOCAB'], nbytes=len(img), period=a.period, params=plist,
                            pw=max_seq.bit_length() - 1, limit=(2_000_000 if a.model == '260k' else 80_000_000) * a.period, **files))
    srcs = [Path(a.top_rtl) if (a.top_rtl and s == 'top_decode.sv') else ROOT / 'rtl/decode' / s for s in SRCS]
    rc, out = compile_and_run(srcs, 'e2e_tb', tb, defines=tuple(a.define))
    line, kv = parse_summary(out)
    step_lines = [x for x in out.splitlines() if x.startswith('STEP')]
    cyc = [int(x.split('cycles=')[1]) for x in step_lines]
    step_names = ['embed', 'seek_layers', 'rms_att', 'quant_att', 'wq', 'wk', 'wv', 'rope_q', 'rope_k', 'kv_store',
                  'attention', 'quant_o', 'wo', 'add_o', 'rms_ffn', 'quant_ffn', 'w1', 'w3', 'swiglu', 'quant_h', 'w2',
                  'add_ffn', 'rms_final', 'quant_final', 'seek_cls', 'classifier', 'done']
    prof = {step_names[int(x.split()[1])]: int(x.split()[2]) for x in out.splitlines() if x.startswith('PROF')}
    o = {'status': 'pass' if rc == 0 and kv.get('errors') == '0' and kv.get('token_errors') == '0' and len(cyc) == len(steps) else 'fail',
         'model': 'stories260K' if a.model == '260k' else 'stories15M', 'max_seq': max_seq, 'kv_bits': a.kv_bits, 'flash_bytes_per_cycle': 1 / a.period, 'prompts': prompts,
         'generated_tokens_per_prompt': a.gen, 'steps': len(steps), 'logits_compared': len(steps) * params['VOCAB'],
         'logit_mismatches': int(kv.get('errors', -1)), 'token_mismatches': int(kv.get('token_errors', -1)),
         'image_bytes': len(img), 'cycles_per_token_mean': (sum(cyc) / len(cyc)) if cyc else None,
         'cycles_per_token_max': max(cyc) if cyc else None,
         'stream_bytes_per_token': params['IMAGE_BYTES'] - params['LAYERS_BASE'] + params['EMB_SCALE_BASE'] + params['CLS_KB'] + 5 * params['CLS_G'],
         'busy_cycles_by_step': prof,
         'golden_tokens': [s[1] for s in seqs], 'first_mismatches': [x for x in out.splitlines() if x.startswith(('MISMATCH', 'TOKEN'))][:8],
         'vvp_command': f'{VVP} build/decode/e2e_tb.vvp', 'defines': a.define, 'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
         'summary': line or out[-500:], 'wall_seconds': round(time.time() - t0, 1)}
    path = a.output or str(ROOT / f'bench/results/decode_e2e_rtl_{tag}.json')
    open(path, 'w').write(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in ('status', 'steps', 'logit_mismatches', 'token_mismatches', 'cycles_per_token_mean', 'first_mismatches', 'summary', 'wall_seconds')}))


if __name__ == '__main__':
    main()
