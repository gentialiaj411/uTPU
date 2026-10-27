"""M5 simulation: host UART -> boolean_core (UART front end + top_decode + qspi_reader) -> flash model (stories260K).

Drives the board's UART protocol (docs/r3_progress.md, M5) from a testbench and checks it against the integer golden
(RunQIntegerLlama via run_decode_rtl.golden):
  * link check, flash JEDEC ID (9Fh), CR1 read; with the flash model's QUAD bit powered up as 0: WREN + WRR to set
    CR1[1], poll SR1 until WIP clears, read CR1 back (the board's own Quad-enable procedure)
  * a truncated frame is dropped after the timeout and the next frame works
  * decode steps (prompt then generated tokens, KV8): argmax token, one probed logit per step, and the cycle count
    reported by the board (vs the testbench's own count) are checked; the weight image sits at flash offset 0x280000
  * negative run: the same decode request with QUAD = 0 must trip the flash model's Quad-read check
  * mutation run (-DBOARD_MUTATE: token bytes swapped in the response) must be caught
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'decode'))
import numpy as np
from harness import ROOT, BUILD, write_mem, parse_summary
from sim import compile_and_run, VVP
import run_decode_rtl as rd
from pack_decode_image import pack

BASE = 0x280000
CPB = 8
TMO = 3000

TB = r"""`timescale 1ns/1ps
module uart_tb;
  localparam int NS = %(ns)d, V = %(vocab)d, NB = %(nbytes)d, BASE = %(base)d, CPB = %(cpb)d, MODE = %(mode)d;
  localparam realtime BIT = 2.0 * CPB;
  logic clk = 0, rst = 1, uart_rx = 1; logic uart_tx;
  always #1 clk = ~clk;
  wire sclk, cs_n; wire [3:0] io_o, io_oe, f_o, f_oe; tri [3:0] bus;
  for (genvar i = 0; i < 4; i++) begin : g_bus
    assign bus[i] = io_oe[i] ? io_o[i] : 1'bz;
    assign bus[i] = f_oe[i] ? f_o[i] : 1'bz;
  end
  boolean_core #(.CPB(CPB), .TMO(%(tmo)d), .PT_HALF(2), %(params)s) dut (.clk, .rst, .uart_rx, .uart_tx, .sclk, .cs_n, .io_o, .io_oe, .io_i(bus));
  qspi_flash_model #(.IMG("%(img)s"), .SIZE(BASE + NB), .BASE(BASE), .QUAD_INIT(1'b0), .ALLOW_REG_CMDS(1'b1),
      .TV(0.765), .THO(0.2), .TCS(1.0)) flash (.sck(sclk), .cs_n, .host_o(io_o), .host_oe(io_oe), .flash_o(f_o), .flash_oe(f_oe));
  logic [47:0] step_mem [NS]; logic [31:0] exp_logits [NS*V];
  integer errors = 0, checks = 0, wr = 0, rd = 0, cycles = 0, shown = 0, nwrr_ok = 0;
  logic [7:0] rxq [4096];
  logic [31:0] rng = 32'h0bad_cafe;
  function automatic int rnd(input int n); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return int'(rng %% 32'(n)); endfunction
  always @(posedge clk) cycles++;
  // UART receiver on uart_tx
  always begin : rx_proc
    logic [7:0] b;
    @(negedge uart_tx); #(BIT / 2.0);
    if (uart_tx === 1'b0) begin
      for (int i = 0; i < 8; i++) begin #(BIT) b[i] = uart_tx; end
      #(BIT); if (uart_tx !== 1'b1) begin errors++; $display("MISMATCH uart framing error"); end
      rxq[wr %% 4096] = b; wr++;
    end
  end
  task automatic send(input [7:0] b);
    uart_rx = 0; #(BIT);
    for (int i = 0; i < 8; i++) begin uart_rx = b[i]; #(BIT); end
    uart_rx = 1; #(BIT);
  endtask
  task automatic expect_byte(input [7:0] e, input string what);
    int guard; guard = 0;
    while (rd >= wr && guard < 200000000) begin @(posedge clk); guard++; end
    if (rd >= wr) begin errors++; $display("MISMATCH timeout waiting for %%s", what); $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end
    checks++;
    if (rxq[rd %% 4096] !== e) begin errors++; if (shown < 12) begin shown++; $display("MISMATCH %%s got=%%h exp=%%h", what, rxq[rd %% 4096], e); end end
    rd++;
    if (errors != 0 && %(stop)d) begin $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end
  endtask
  task automatic spi(input int ntx, input int nrx, input [63:0] tx, input [63:0] exp_rx, input string what);   // tx[63:56] first
    send(8'h53); send(ntx); send(nrx);
    for (int i = 0; i < ntx; i++) send(tx[63 - 8*i -: 8]);
    expect_byte(8'h73, what); expect_byte(nrx, what);
    for (int i = 0; i < nrx; i++) expect_byte(exp_rx[63 - 8*i -: 8], what);
  endtask
  task automatic step(input int s, input int idx);
    int t0; logic [31:0] ev; logic [31:0] cyc_rep; logic [7:0] cb [4];
    t0 = cycles;
    send(8'h54); send(step_mem[s][39:32]); send(step_mem[s][47:40]); send(step_mem[s][7:0]); send(idx[7:0]); send(idx[15:8]);
    expect_byte(8'h52, "R tag");
    expect_byte(step_mem[s][23:16], "token lo"); expect_byte(step_mem[s][31:24], "token hi");
    ev = (idx < V) ? exp_logits[s*V + idx] : 32'd0;
    for (int i = 0; i < 4; i++) begin
      while (rd >= wr) @(posedge clk);
      cb[i] = rxq[rd %% 4096]; rd++;
    end
    cyc_rep = {cb[3], cb[2], cb[1], cb[0]};
    expect_byte(ev[7:0], "logit0"); expect_byte(ev[15:8], "logit1"); expect_byte(ev[23:16], "logit2"); expect_byte(ev[31:24], "logit3");
    checks++;
    $display("STEP %%0d idx=%%0d token=%%0d cycles_reported=%%0d wall_cycles=%%0d", s, idx, step_mem[s][23:16] | (step_mem[s][31:24] << 8), cyc_rep, cycles - t0);
    if (cyc_rep < 100000 || cyc_rep > cycles - t0) begin errors++; $display("MISMATCH implausible cycle count %%0d", cyc_rep); end
  endtask
  initial begin
    $readmemh("%(steps)s", step_mem); $readmemh("%(logits)s", exp_logits);
    repeat (4) @(posedge clk); rst = 0; repeat (50) @(posedge clk);
    if (MODE == 0) begin
      // negative: QUAD bit is 0, a decode request must trip the flash model's check (this sim is expected to end in $fatal)
      send(8'h54); send(1); send(0); send(0); send(0); send(0);
      repeat (200000) @(posedge clk);
      $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d NO_FATAL", errors, checks, cycles);
      $finish;
    end
    send(8'h56); expect_byte(8'h76, "V tag"); expect_byte(8'h01, "V ver");
    spi(1, 6, {8'h9F, 56'd0}, {8'h01, 8'h20, 8'h18, 8'h4D, 8'h00, 8'h80, 16'd0}, "RDID");
    spi(1, 1, {8'h35, 56'd0}, {8'h00, 56'd0}, "RDCR before");
    send(8'h54); send(8'h01); send(8'h00);                        // truncated frame, must time out
    repeat (TMO_CYC) @(posedge clk);
    send(8'h56); expect_byte(8'h76, "V after timeout"); expect_byte(8'h01, "V ver 2");
    spi(1, 0, {8'h06, 56'd0}, 64'd0, "WREN");
    spi(1, 1, {8'h05, 56'd0}, {8'h02, 56'd0}, "RDSR wel set");
    spi(3, 0, {8'h01, 8'h00, 8'h02, 40'd0}, 64'd0, "WRR");
    spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip");
    spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip 2");
    spi(1, 1, {8'h05, 56'd0}, {8'h00, 56'd0}, "RDSR done");
    spi(1, 1, {8'h35, 56'd0}, {8'h02, 56'd0}, "RDCR after");
    for (int s = 0; s < NS; s++) step(s, (s == 1) ? V + 5 : rnd(V));
    $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d ncmds=%%0d nwrr=%%0d", errors, checks, cycles, flash.ncmds, flash.n_wrr);
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=int, default=1)
    ap.add_argument('--gen', type=int, default=2)
    ap.add_argument('--output', default=str(ROOT / 'bench/results/boolean_uart_rtl.json'))
    a = ap.parse_args()
    t0 = time.time()
    prompts = [[1, 403, 260], [1, 87]][:a.prompts]
    exp, seqs = rd.golden('260k', 8, prompts, a.gen)
    img, params = pack(exp)
    steps, logits = [], []
    for tokens, outs, lg in seqs:
        for p, (t, o, l) in enumerate(zip(tokens, outs, lg)):
            steps.append((t << 32) | (o << 16) | p)
            logits += [int(x) for x in l]
    files = dict(img=write_mem('uart_img_260k', list(img), 8), steps=write_mem('uart_steps', steps, 48), logits=write_mem('uart_logits', logits, 32))
    keys = ['D', 'H', 'NL', 'NH', 'NKV', 'HS', 'VOCAB', 'CLS4', 'WQ4', 'WK4', 'WV4', 'WO4', 'W14', 'W34', 'W24',
            'CLS_BASE', 'EMB_SCALE_BASE', 'LAYERS_BASE', 'IMAGE_BYTES', 'CLS_TILE_BYTES', 'CLS_KB', 'CLS_G', 'RECIP_D', 'NORM_MULT',
            'EPS', 'GFB', 'NORM_SHIFT', 'SILU_MULT', 'SWIGLU_MULT', 'SCORE_MULT', 'SILU_SHIFT', 'SWIGLU_SHIFT', 'SCORE_SHIFT']
    plist = ', '.join(f'.{k}({params[k]})' for k in keys) + ', .MAX_SEQ(64), .KV4(0), .ROPE_CS("rtl/decode/luts/rope_cs_q15_h8.mem"), .ROPE_DEPTH(2048)'
    srcs = [ROOT / 'rtl/decode' / s for s in rd.SRCS] + [ROOT / 'rtl/decode/qspi_reader.sv'] + \
           [ROOT / 'rtl/board' / s for s in ('uart_rx.sv', 'uart_tx.sv', 'board_ctrl.sv', 'boolean_core.sv')] + [ROOT / 'tools/decode/tb/qspi_flash_model.sv']

    def build(mode, stop, name):
        tb = BUILD / f'uart_{name}_tb.sv'
        text = TB % dict(ns=len(steps), vocab=params['VOCAB'], nbytes=len(img), base=BASE, cpb=CPB, mode=mode, tmo=TMO, params=plist, stop=stop, **files)
        tb.write_text(text.replace('TMO_CYC', str(3 * TMO)).replace('.QUAD_INIT(MODE == 0 ? 1\'b0 : 1\'b0)', '.QUAD_INIT(1\'b0)'))
        return tb

    # main run (QUAD powers up 0; the TB enables it through the UART SPI passthrough)
    tb = build(1, 0, 'main')
    rc, out = compile_and_run(srcs, 'uart_tb', tb)
    line, kv = parse_summary(out)
    # mutation: swapped token bytes in the response must be caught (stop at the first error)
    tbm = build(1, 1, 'mut')
    mrc, mout = compile_and_run(srcs, 'uart_tb', tbm, defines=('BOARD_MUTATE',))
    mline, mkv = parse_summary(mout)
    # negative: QUAD = 0 and a decode request -> the flash model must $fatal
    tbn = build(0, 0, 'neg')
    nrc, nout = compile_and_run(srcs, 'uart_tb', tbn)
    neg_caught = 'Quad Output Read issued with CR1.QUAD = 0' in nout
    step_lines = [x for x in out.splitlines() if x.startswith('STEP')]
    o = {'status': 'pass' if rc == 0 and kv.get('errors') == '0' and len(step_lines) == len(steps) and int(kv.get('nwrr', 0)) >= 1
         and int(mkv.get('errors', 0)) > 0 and neg_caught else 'fail',
         'model': 'stories260K (boolean_core with 260K parameters; the 15M configuration is the default of the same RTL)',
         'uart_cpb': CPB, 'flash_base': BASE, 'steps': len(steps), 'prompts': prompts, 'golden_tokens': [s[1] for s in seqs],
         'checks': int(kv.get('checks', -1)), 'mismatch_count': int(kv.get('errors', -1)), 'steps_log': step_lines,
         'quad_enable_via_uart': {'flash_model_QUAD_powerup': 0, 'wrr_commands_applied': int(kv.get('nwrr', 0))},
         'mutation_mismatch_count': int(mkv.get('errors', -1)), 'mutation_expected': '>0 (BOARD_MUTATE swaps the response token bytes)',
         'negative_quad0_read_trips_flash_model': neg_caught, 'negative_stdout_tail': [x for x in nout.splitlines() if 'FLASH' in x][:2],
         'first_mismatches': [x for x in out.splitlines() if x.startswith('MISMATCH')][:8],
         'vvp_command': f'{VVP} build/decode/uart_main_tb.vvp', 'summary': line, 'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
         'wall_seconds': round(time.time() - t0, 1)}
    Path(a.output).write_text(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in ('status', 'steps', 'checks', 'mismatch_count', 'mutation_mismatch_count', 'negative_quad0_read_trips_flash_model', 'summary', 'wall_seconds')}))


if __name__ == '__main__':
    main()
