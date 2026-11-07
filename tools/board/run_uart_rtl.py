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
from harness import ROOT, BUILD, write_mem, parse_summary, finish, provenance, ARTIFACT_SOURCES
from sim import compile_and_run, VVP
import run_decode_rtl as rd
from pack_decode_image import pack
sys.path.insert(0, str(Path(__file__).resolve().parent))
import boolean_host as bh

# --- interface names shared with the RTL / flash-model changes (DDR contract C1/C3); integration fixes a mismatch HERE only ---
BOARD_DDR_PARAM = 'DDR'          # boolean_core / boolean_top parameter; 'V' reply is 1 for DDR=0 and 2 for DDR=1
MODEL_CR1_PARAM = 'LC_INIT/QUAD_INIT'   # qspi_flash_model power-up CR1 fields (--ddr starts at LC 00, QUAD 0; host sets QUAD and leaves LC=00)
CR1_WANT_SDR, CR1_WANT_DDR = 0x02, 0x02
BASE = 0x280000
CPB = 8
TMO = 3000

TB = r"""`timescale 1ns/1ps
module uart_tb;
  localparam int NS = %(ns)d, V = %(vocab)d, NB = %(nbytes)d, BASE = %(base)d, CPB = %(cpb)d, MODE = %(mode)d;
  localparam logic [7:0] VER = 8'h%(ver)02x, CR1_WANT = 8'h%(cr1want)02x;   // 'V' reply of this build; CR1 the host procedure must leave in the flash
  localparam realtime BIT = 2.0 * CPB;
  logic clk = 0, rst = 1, uart_rx = 1; logic uart_tx;
  always #1 clk = ~clk;
  wire pad_sclk, pad_cs_n; wire [3:0] pad_io_o, pad_io_t, f_o, f_oe; tri [3:0] bus;
  for (genvar i = 0; i < 4; i++) begin : g_bus
    assign bus[i] = !pad_io_t[i] ? pad_io_o[i] : 1'bz;
    assign bus[i] = f_oe[i] ? f_o[i] : 1'bz;
  end
  boolean_core #(.CPB(CPB), .TMO(%(tmo)d), .WDOG(%(wdog)d), .PT_HALF(2), %(params)s) dut (.clk, .rst, .uart_rx, .uart_tx, .pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i(bus));
  qspi_flash_model #(.IMG("%(img)s"), .SIZE(BASE + NB), .BASE(BASE), .QUAD_INIT(1'b0)%(mparams)s, .ALLOW_REG_CMDS(1'b1),
      .TV(0.765), .THO(0.2), .TCS(1.0), .TCSS(0.3), .TCSH(0.3), .TSU(0.3), .THD(0.2)) flash (.sck(pad_sclk), .cs_n(pad_cs_n), .host_o(pad_io_o), .host_oe(~pad_io_t), .flash_o(f_o), .flash_oe(f_oe));
  logic [47:0] step_mem [NS]; logic [31:0] exp_logits [NS*V];
  integer errors = 0, checks = 0, wr = 0, rd = 0, cycles = 0, shown = 0, nwrr_ok = 0;
  logic [7:0] rxq [4096];
  logic [31:0] rng = 32'h0bad_cafe;
  function automatic int rnd(input int n); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return int'(rng %% 32'(n)); endfunction
  integer t_start = 0, t_done = 0, n_starts = 0, exp_steps = 0; logic [7:0] exp_pos = 0;
  always @(posedge clk) begin cycles++; if (dut.dec_start) t_start = cycles; if (dut.dec_done) t_done = cycles; end
  // full-width position path: the pos byte of the request must arrive at the top_decode port unchanged (PW = 7 here)
  always @(posedge clk) if (dut.dec_start) begin
    n_starts++; checks++;
    if (dut.dec_pos !== exp_pos[6:0] || exp_pos[7]) begin errors++; $display("MISMATCH dec_pos at top_decode port %%0d, request %%0d", dut.dec_pos, exp_pos); end
    if (dut.u_dec.pos !== exp_pos[6:0]) begin errors++; $display("MISMATCH top_decode pos input %%0d, request %%0d", dut.u_dec.pos, exp_pos); end
  end
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
    while (rd >= wr && guard < 4000000) begin @(posedge clk); #0.1; guard++; end
    if (rd >= wr) begin errors++; $display("MISMATCH timeout waiting for %%s", what); $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end
    checks++;
    if (rxq[rd %% 4096] !== e) begin errors++; if (shown < 12) begin shown++; $display("MISMATCH %%s got=%%h exp=%%h", what, rxq[rd %% 4096], e); end end
    rd++;
    if (errors != 0 && %(stop)d) begin $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end
  endtask
  task automatic send_bad_stop(input [7:0] b);   // framing error: stop bit low for one bit time, then idle
    uart_rx = 0; #(BIT);
    for (int i = 0; i < 8; i++) begin uart_rx = b[i]; #(BIT); end
    uart_rx = 0; #(BIT); uart_rx = 1; #(BIT);
  endtask
  task automatic expect_idle(input int ncyc, input string what);   // no response byte may arrive
    repeat (ncyc) @(posedge clk); #0.1;
    checks++;
    if (rd != wr) begin errors++; $display("MISMATCH unexpected response byte(s) after %%s: %%0d pending, first %%h", what, wr - rd, rxq[rd %% 4096]);
      if (%(stop)d) begin $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end end
  endtask
  task automatic expect_err(input [7:0] code, input int next, input string what);
    expect_byte(8'h45, {what, " E tag"}); expect_byte(code, {what, " E code"}); expect_byte(8'(next), {what, " E next"});
    expect_idle(3000, what);
  endtask
  task automatic doreset;
    @(negedge clk); rst = 1; repeat (4) @(negedge clk); rst = 0; repeat (8) @(negedge clk);
  endtask
  task automatic ping(input string what);
    send(8'h56); expect_byte(8'h76, {what, " v"}); expect_byte(VER, {what, " v1"});
  endtask
  task automatic rdid(input string what);
    spi(1, 6, {8'h9F, 56'd0}, {8'h01, 8'h20, 8'h18, 8'h4D, 8'h00, 8'h80, 16'd0}, what);
  endtask
  task automatic spi(input int ntx, input int nrx, input [63:0] tx, input [63:0] exp_rx, input string what);   // tx[63:56] first
    send(8'h53); send(ntx); send(nrx);
    for (int i = 0; i < ntx; i++) send(tx[63 - 8*i -: 8]);
    expect_byte(8'h73, what); expect_byte(nrx, what);
    for (int i = 0; i < nrx; i++) expect_byte(exp_rx[63 - 8*i -: 8], what);
  endtask
  task automatic step(input int s, input int idx, input bit extra);   // extra: a second 'T' frame is sent while the step runs (must be ignored)
    int t0; logic [31:0] ev; logic [31:0] cyc_rep; logic [7:0] cb [4];
    t0 = cycles; exp_pos = step_mem[s][7:0]; exp_steps++;
    send(8'h54); send(step_mem[s][39:32]); send(step_mem[s][47:40]); send(step_mem[s][7:0]); send(idx[7:0]); send(idx[15:8]);
    if (extra) begin send(8'h54); send(8'h09); send(8'h00); send(8'h02); send(8'h00); send(8'h00); end
    expect_byte(8'h52, "R tag");
    expect_byte(step_mem[s][23:16], "token lo"); expect_byte(step_mem[s][31:24], "token hi");
    ev = (idx < V) ? exp_logits[s*V + idx] : 32'd0;
    for (int i = 0; i < 4; i++) begin
      while (rd >= wr) begin @(posedge clk); #0.1; end
      cb[i] = rxq[rd %% 4096]; rd++;
    end
    cyc_rep = {cb[3], cb[2], cb[1], cb[0]};
    expect_byte(ev[7:0], "logit0"); expect_byte(ev[15:8], "logit1"); expect_byte(ev[23:16], "logit2"); expect_byte(ev[31:24], "logit3");
    checks++;
    $display("STEP %%0d idx=%%0d token=%%0d cycles_reported=%%0d wall_cycles=%%0d", s, idx, step_mem[s][23:16] | (step_mem[s][31:24] << 8), cyc_rep, cycles - t0);
    if (cyc_rep != t_done - t_start + 1) begin errors++; $display("MISMATCH reported cycles %%0d, testbench count %%0d", cyc_rep, t_done - t_start + 1);
      if (%(stop)d) begin $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d", errors, checks, cycles); $finish; end end
    if (extra) expect_idle(2000, "back-to-back T frame during a step");
  endtask
  initial begin
    $readmemh("%(steps)s", step_mem); $readmemh("%(logits)s", exp_logits);
    repeat (4) @(posedge clk); #0.1; rst = 0; repeat (50) @(posedge clk); #0.1;
    if (MODE == 2) begin
      // robustness: watchdog on a stalled byte stream, then reset in the middle of a frame, an SPI read and a decode step
      send(8'h56); expect_byte(8'h76, "V tag"); expect_byte(VER, "V ver");
      spi(1, 1, {8'h35, 56'd0}, {8'h00, 56'd0}, "RDCR");
      spi(1, 0, {8'h06, 56'd0}, 64'd0, "WREN"); spi(3, 0, {8'h01, 8'h00, CR1_WANT, 40'd0}, 64'd0, "WRR");
      spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip"); spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip 2"); spi(1, 1, {8'h05, 56'd0}, {8'h00, 56'd0}, "RDSR done");
      step(0, 3, 0);                                             // pos 0 completes, next = 1
      // stall: the flash byte stream stops (st_valid forced low) after the step has started; the watchdog must fire
      exp_pos = 8'h01; send(8'h54); send(step_mem[1][39:32]); send(step_mem[1][47:40]); send(8'h01); send(8'h00); send(8'h00);
      repeat (20000) @(posedge clk); #0.1; force dut.st_valid = 1'b0;
      expect_err(8'h04, 1, "watchdog");
      release dut.st_valid;
      // the engine and the reader were reset: a retry of pos 1 and the following steps must match the golden
      step(1, 5, 0);
      send(8'h58); expect_byte(8'h78, "x tag"); expect_byte(8'h01, "x ver");
      step(0, 7, 0); step(1, 9, 0);
      // reset in the middle of a UART frame
      send(8'h54); send(8'h01); send(8'h00); doreset; expect_idle(2000, "reset mid frame");
      ping("after reset mid frame"); rdid("RDID after reset mid frame");
      // reset in the middle of an SPI register read (after the opcode: the flash model rejects a CS# rise inside the command byte)
      send(8'h53); send(8'd1); send(8'd6); send(8'h9F);
      repeat (380) @(posedge clk); #0.1; doreset; expect_idle(2000, "reset mid SPI");
      ping("after reset mid SPI"); rdid("RDID after reset mid SPI");
      // reset in the middle of a decode step (data phase of a burst)
      exp_pos = 8'h00; send(8'h54); send(step_mem[0][39:32]); send(step_mem[0][47:40]); send(8'h00); send(8'h00); send(8'h00);
      repeat (300000) @(posedge clk); #0.1; doreset; expect_idle(2000, "reset mid decode step");
      ping("after reset mid decode"); rdid("RDID after reset mid decode");
      step(0, 11, 0); step(1, 13, 0);
      $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d starts=%%0d", errors, checks, cycles, n_starts);
      $finish;
    end
    if (MODE == 3) begin
      // DDR negative: QUAD = 1 but legacy LC = 11. The flash model policy must reject this boot-unsafe configuration.
      send(8'h56); expect_byte(8'h76, "V tag"); expect_byte(VER, "V ver");
      spi(1, 0, {8'h06, 56'd0}, 64'd0, "WREN"); spi(3, 0, {8'h01, 8'h00, 8'hC2, 40'd0}, 64'd0, "WRR LC=11");
      spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip"); spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip 2"); spi(1, 1, {8'h05, 56'd0}, {8'h00, 56'd0}, "RDSR done");
      spi(1, 1, {8'h35, 56'd0}, {8'hC2, 56'd0}, "RDCR LC=11");
      step(0, 3, 0);
      $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d NOT_DETECTED", errors, checks, cycles);
      $finish;
    end
    if (MODE == 0) begin
      // negative: QUAD bit is 0, a decode request must trip the flash model's check (this sim is expected to end in $fatal)
      send(8'h54); send(1); send(0); send(0); send(0); send(0);
      repeat (200000) @(posedge clk); #0.1;
      $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d NO_FATAL", errors, checks, cycles);
      $finish;
    end
    send(8'h56); expect_byte(8'h76, "V tag"); expect_byte(VER, "V ver");
    send_bad_stop(8'h56); expect_idle(2000, "framing error");
    // rejected requests run no decode and answer with E frames; X restarts the sequence
    send(8'h58); expect_byte(8'h78, "x tag"); expect_byte(8'h01, "x ver");
    send(8'h54); send(V[7:0]); send(V[15:8]); send(8'h00); send(8'h00); send(8'h00); expect_err(8'h01, 0, "token == VOCAB");
    send(8'h54); send(8'h01); send(8'h00); send(8'h80); send(8'h00); send(8'h00); expect_err(8'h02, 0, "pos == MAX_SEQ");
    send(8'h54); send(8'h01); send(8'h00); send(8'hFF); send(8'h00); send(8'h00); expect_err(8'h02, 0, "pos 255");
    send(8'h54); send(8'h01); send(8'h00); send(8'h01); send(8'h00); send(8'h00); expect_err(8'h03, 0, "pos 1 before pos 0");
    send(8'h53); send(8'd9); send(8'd1); expect_err(8'h05, 0, "S ntx 9");        // a byte with a bad stop bit must be dropped
    spi(1, 6, {8'h9F, 56'd0}, {8'h01, 8'h20, 8'h18, 8'h4D, 8'h00, 8'h80, 16'd0}, "RDID");
    // the host tool's own register sequence (boolean_host.py: info, enable-quad, info) played against the RTL and the flash model;
    // the expected response bytes come from the host's software stub (an independent Python model of board_ctrl + S25FL128S registers)
%(hostseq)s
    spi(1, 1, {8'h35, 56'd0}, {CR1_WANT, 56'd0}, "RDCR after host enable-quad");
    send(8'h54); send(8'h01); send(8'h00);                        // truncated frame, must time out
    repeat (TMO_CYC) @(posedge clk); #0.1;
    send(8'h56); expect_byte(8'h76, "V after timeout"); expect_byte(VER, "V ver 2");
    spi(1, 0, {8'h06, 56'd0}, 64'd0, "WREN");
    spi(1, 1, {8'h05, 56'd0}, {8'h02, 56'd0}, "RDSR wel set");
    spi(3, 0, {8'h01, 8'h00, CR1_WANT, 40'd0}, 64'd0, "WRR");
    spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip");
    spi(1, 1, {8'h05, 56'd0}, {8'h01, 56'd0}, "RDSR wip 2");
    spi(1, 1, {8'h05, 56'd0}, {8'h00, 56'd0}, "RDSR done");
    spi(1, 1, {8'h35, 56'd0}, {CR1_WANT, 56'd0}, "RDCR after");
    for (int s = 0; s < NS; s++) begin
      if (s > 0) spi(1, 1, {8'h35, 56'd0}, {CR1_WANT, 56'd0}, "RDCR between decode steps");   // SPI access with the reader parked mid-burst
      if (s == 1) begin    // state-changing commands need CS# to rise after each S frame, even though the reader has CS# low (parked burst)
        spi(1, 0, {8'h06, 56'd0}, 64'd0, "WREN between steps");
        spi(1, 1, {8'h05, 56'd0}, {8'h02, 56'd0}, "RDSR WEL between steps");
        spi(1, 0, {8'h04, 56'd0}, 64'd0, "WRDI between steps");
        spi(1, 1, {8'h05, 56'd0}, {8'h00, 56'd0}, "RDSR WEL cleared between steps");
      end
      if (s == 1) begin send(8'h54); send(8'h01); send(8'h00); send(8'h05); send(8'h00); send(8'h00); expect_err(8'h03, 1, "pos 5 with next 1"); end
      step(s, (s == 1) ? V + 5 : rnd(V), s == 0);          // step 0 also gets a second 'T' frame while busy
    end
    checks++; if (n_starts != NS) begin errors++; $display("MISMATCH %%0d decode starts, expected %%0d (a rejected request ran a decode?)", n_starts, NS); end
    $display("SUMMARY errors=%%0d checks=%%0d cycles=%%0d ncmds=%%0d nwrr=%%0d", errors, checks, cycles, flash.ncmds, flash.n_wrr);
    $finish;
  end
endmodule
"""


class _Rec(bh.Stub):
    def __init__(self):
        super().__init__(); self.pairs = []
    def write(self, b):
        n0 = len(self.buf); super().write(b); self.pairs.append((bytes(b), self.buf[n0:]))


def host_sequence(version=1):
    """(frame, expected response) pairs produced by boolean_host.Board for info -> enable-quad -> info against the QUAD=0, LC=00 power-up state
    (version 1: the SDR build's LC=00 target; version 2: the DDR build, enable-quad must leave LC=00)."""
    r = _Rec(); r.version = version; b = bh.Board(r)
    i0 = b.info(); assert not i0['compatible'] and b.enable_quad()[0] == 'set'; i1 = b.info(); assert i1['compatible']
    assert r.cr1 == (CR1_WANT_DDR if version == 2 else CR1_WANT_SDR), r.cr1
    lines = []
    for k, (f, resp) in enumerate(r.pairs):
        lines.append('    ' + ' '.join(f"send(8'h{x:02x});" for x in f) + ' ' + ' '.join(f"expect_byte(8'h{x:02x}, \"host seq {k}\");" for x in resp))
    return chr(10).join(lines), len(r.pairs), r.n_wrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=int, default=1)
    ap.add_argument('--gen', type=int, default=2)
    ap.add_argument('--output', default=None, help='default bench/results/boolean_uart_rtl.json, or boolean_uart_ddr_rtl.json with --ddr')
    ap.add_argument('--ddr', action='store_true', help='DDR build: boolean_core DDR=1 (V reply 2), flash powers up LC=00/QUAD=0 and host enables QUAD while leaving LC=00')
    ap.add_argument('--fast', action='store_true', help='fault-injection mode: main run only, stop at the first error')
    ap.add_argument('--replace', action='append', default=[], help='ORIG=MUTANT source file substitution (fault injection)')
    ap.add_argument('--sim', choices=('icarus','verilator'), default=__import__('sim').SIMULATOR, help='simulation backend (default icarus)')
    a = ap.parse_args()
    ddr = a.ddr
    if a.output is None:
        a.output = str(ROOT / ('bench/results/boolean_uart_ddr_rtl.json' if ddr else 'bench/results/boolean_uart_rtl.json'))
    ver, cr1want = (2, CR1_WANT_DDR) if ddr else (1, CR1_WANT_SDR)
    pfx = 'ddr_' if ddr else ''
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
    plist = ', '.join(f'.{k}({params[k]})' for k in keys) + ', .MAX_SEQ(128), .KV4(0), .ROPE_CS("rtl/decode/luts/rope_cs_q15_h8.mem"), .ROPE_DEPTH(2048)' + (f', .{BOARD_DDR_PARAM}(1)' if ddr else '')
    srcs = [ROOT / 'rtl/decode' / s for s in rd.SRCS] + [ROOT / 'rtl/decode/qspi_reader.sv'] + \
           [ROOT / 'rtl/board' / s for s in ('uart_rx.sv', 'uart_tx.sv', 'qspi_pad_regs.sv', 'board_ctrl.sv', 'boolean_core.sv')] + [ROOT / 'tools/decode/tb/qspi_flash_model.sv']

    ctrl_idx = [Path(x).name for x in srcs].index('board_ctrl.sv')
    for r in a.replace:
        orig, mut = r.split('=')
        srcs = [Path(mut) if Path(s).resolve() == (ROOT / orig).resolve() else s for s in srcs]

    hostseq, n_host_frames, n_host_wrr = host_sequence(ver)

    def build(mode, stop, name, wdog=268435456):
        tb = BUILD / f'uart_{pfx}{name}_tb.sv'
        # the robustness run resets the FPGA at an arbitrary cycle of a DDR burst (it may cut a byte; the datasheet allows a read
        # to end anywhere in data output); every other run keeps the model's "reader stops only between bytes" check
        abort = ", .ALLOW_DATA_ABORT(1'b1)" if (ddr and name.startswith('robust')) else ''
        text = TB % dict(ver=ver, cr1want=cr1want, mparams=(", .LC_INIT(2'b00), .DTV(0.65), .DTHO(0.15), .DTSU(0.2), .DTHD(0.2)" + abort if ddr else ''), ns=len(steps), vocab=params['VOCAB'], nbytes=len(img), base=BASE, cpb=CPB, mode=mode, tmo=TMO, params=plist, stop=stop, wdog=wdog, hostseq=hostseq, **files)
        tb.write_text(text.replace('TMO_CYC', str(3 * TMO)).replace('.QUAD_INIT(MODE == 0 ? 1\'b0 : 1\'b0)', '.QUAD_INIT(1\'b0)'))
        return tb

    def unit_tb():
        ut = ROOT / 'tools/board/tb/board_ctrl_unit_tb.sv'
        urc, uout = compile_and_run([srcs[ctrl_idx]], 'board_ctrl_unit_tb', ut)
        uline, ukv = parse_summary(uout)
        return urc, uout, uline, ukv

    if a.fast:
        # cheapest detector first: board_ctrl against a stub decoder (position width, bounds, sequence, watchdog, resets); the slower
        # full-system runs only when the unit testbench passes
        urc, uout, uline, ukv = unit_tb()
        uerrs = (int(ukv['errors']) if 'errors' in ukv else 0) + (1 if urc != 0 or 'errors' not in ukv else 0)
        errs, rerrs, out, rout, line = 0, 0, '', '', 'skipped (unit testbench already detected)'
        if uerrs == 0:
            tb = build(1, 1, 'fast')
            rc, out = compile_and_run(srcs, 'uart_tb', tb)
            line, kv = parse_summary(out)
            errs = (int(kv['errors']) if 'errors' in kv else 0) + (1 if rc != 0 or 'errors' not in kv else 0)   # a flash-model $fatal counts as a detection
        if uerrs + errs == 0 and any(Path(r.split('=')[0]).name == 'boolean_core.sv' for r in a.replace):    # wiring faults need the real engine and a stalled stream
            rrc, rout = compile_and_run(srcs, 'uart_tb', build(2, 1, 'robust_fast', wdog=1400000))
            rl, rk = parse_summary(rout)
            rerrs = (int(rk['errors']) if 'errors' in rk else 0) + (1 if rrc != 0 or 'errors' not in rk else 0)
        o = {'mode': 'fast', 'status': 'pass' if errs + uerrs + rerrs == 0 else 'fail', 'mismatch_count': errs + uerrs + rerrs, 'real_mismatch_count': errs + uerrs + rerrs, 'summary': line, 'unit_summary': uline,
             'first_mismatches': [x for x in (out + uout + rout).splitlines() if x.startswith(('MISMATCH', 'FATAL'))][:4]}
        finish(o, a.output, tuple(o))   # exit 1 when a mismatch was seen (fault injection expects that for a detected mutant)
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
    # SDR: the 6Bh check; DDR: the model's check on the EDh read (its exact wording is the flash-model change's, so accept any FLASH line naming CR1.QUAD = 0)
    neg_caught = ('Quad Output Read issued with CR1.QUAD = 0' in nout) if not ddr else any('FLASH' in x and 'CR1.QUAD = 0' in x for x in nout.splitlines())
    lc_neg_caught, lc_neg_line = None, None
    if ddr:   # QUAD = 1 but LC = 00 under the DDR build must be caught (flash model fatal, or golden mismatch)
        lrc, lout = compile_and_run(srcs, 'uart_tb', build(3, 1, 'neg_lc'))
        lc_neg_line, lkv = parse_summary(lout)
        lc_neg_caught = 'NOT_DETECTED' not in lout and (lrc != 0 or int(lkv.get('errors', 0)) > 0 or 'errors' not in lkv)
    # robustness run: watchdog on a stalled byte stream + recovery, resets mid frame / SPI read / decode step
    tbr = build(2, 0, 'robust', wdog=1400000)
    rrc, rout = compile_and_run(srcs, 'uart_tb', tbr)
    rline, rkv = parse_summary(rout)
    robust_ok = rrc == 0 and rkv.get('errors') == '0'
    urc, uout, uline, ukv = unit_tb()
    unit_ok = urc == 0 and ukv.get('errors') == '0' and int(ukv.get('checks', 0)) > 0
    step_lines = [x for x in out.splitlines() if x.startswith('STEP')]
    o = {'status': 'pass' if rc == 0 and kv.get('errors') == '0' and len(step_lines) == len(steps) and int(kv.get('nwrr', 0)) >= 1 + n_host_wrr
         and int(mkv.get('errors', 0)) > 0 and neg_caught and robust_ok and unit_ok and (not ddr or lc_neg_caught) else 'fail',
         'model': 'stories260K (boolean_core with 260K parameters; the 15M configuration is the default of the same RTL)',
         'uart_cpb': CPB, 'flash_base': BASE, 'steps': len(steps), 'prompts': prompts, 'golden_tokens': [s[1] for s in seqs],
         'checks': int(kv.get('checks', -1)), 'mismatch_count': int(kv.get('errors', -1)), 'steps_log': step_lines,
         'robustness_run': {'summary': rline, 'ok': robust_ok, 'steps_in_run': len([x for x in rout.splitlines() if x.startswith('STEP')]),
                            'covers': 'watchdog (stalled byte stream) + retry, X, reset mid UART frame / mid SPI read / mid decode step, each followed by golden-checked steps', 'first_mismatches': [x for x in rout.splitlines() if x.startswith(('MISMATCH', 'FATAL'))][:6]},
         'board_ctrl_unit': {'summary': uline, 'ok': unit_ok, 'covers': 'pos 0..127 reaches the decode port (PW=7), bounds, sequence rule, S errors, watchdog + abort pulse, resets', 'first_mismatches': [x for x in uout.splitlines() if x.startswith('MISMATCH')][:6]},
         'max_seq': 128, 'position_width_bits': 7,
         'host_sequence_frames': n_host_frames, 'quad_enable_via_uart': {'flash_model_QUAD_powerup': 0, 'host_tool_wrr_expected': n_host_wrr, 'wrr_commands_applied': int(kv.get('nwrr', 0))},
         'mutation_mismatch_count': int(mkv.get('errors', -1)), 'mutation_expected': '>0 (BOARD_MUTATE swaps the response token bytes)',
         **({'ddr': 1, 'build_version': ver, 'flash_model_cr1_powerup': 0, 'flash_cr1_after_host_procedure': cr1want, 'board_ddr_param': BOARD_DDR_PARAM, 'model_cr1_param': MODEL_CR1_PARAM,
             'negative_lc00_under_ddr_build_detected': lc_neg_caught, 'negative_lc00_summary': lc_neg_line} if ddr else {}),
         'negative_quad0_read_trips_flash_model': neg_caught, 'negative_stdout_tail': [x for x in nout.splitlines() if 'FLASH' in x][:2],
         'first_mismatches': [x for x in out.splitlines() if x.startswith('MISMATCH')][:8],
         'vvp_command': f'{VVP} build/decode/uart_main_tb.vvp', 'summary': line, 'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
         'wall_seconds': round(time.time() - t0, 1),
         'provenance': provenance(ARTIFACT_SOURCES['boolean_uart_ddr_rtl' if ddr else 'boolean_uart_rtl'])}
    finish(o, a.output, ('status', 'steps', 'checks', 'mismatch_count', 'mutation_mismatch_count', 'negative_quad0_read_trips_flash_model', 'summary', 'wall_seconds'))


if __name__ == '__main__':
    main()
