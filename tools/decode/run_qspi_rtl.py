"""Unit test: rtl/decode/qspi_reader.sv against the independent flash model (tools/decode/tb/qspi_flash_model.sv).

Two kinds of run per reader configuration:
  image : directed + randomized cases over a 16 KiB random image (seek to 0 / last byte / every alignment, back-to-back
          seeks, seeks mid-byte, long backpressure stalls, a seek in the same cycle as a byte handoff);
  sparse: the model holds no image, every byte of the 2^24 (ADDR_BYTES=3) or 2^32 (ADDR_BYTES=4) space is a hash of its
          address (model: sparse_byte; checker: exp_byte, written separately). Seeks walk every address bit, hit the board
          image base 0x280000, the top of the 16 MiB flash and the 24-bit boundary (audit N01).
Every delivered byte is compared with the expected byte. A -DQSPI_MUTATE build (nibble swap) must be caught on exactly the
bytes whose nibbles differ. All configurations run behind startupe2_cclk_shim, which swallows the first USRCCLKO cycles the
way STARTUPE2 does (audit N02). Parameter combinations: every HALF 1..4 x ADDR_BYTES 3/4 x SAMPLE_LATE 0/1 (direct pads) and
HALF 1..4 behind the pad registers with CAP_DLY=2 (audit N04); qspi_reader rejects any other value at elaboration, which the
'rejected' section here confirms. 'negative' runs are model/reader disagreements (latency code, 4-byte address mode,
STARTUP_PULSES vs swallowed edges) that must be DETECTED.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from harness import ROOT, BUILD, write_mem, parse_summary, provenance, ARTIFACT_SOURCES, finish
from sim import compile_and_run, VVP

NB = 16384
PADS_SLOWIN = dict(pads=1, sck_d=7.5, out_d=4.0, in_d=2.0, tv=8.0)
# direct pads (no pad registers); then behind qspi_pad_regs with CAP_DLY=2 and board/flash delay corners:
#   slowin : CCLK delay 7.5 (DS189 TUSRCCLKO max), flash tV 8.0 (max), 2.0 ns input path  -> worst input setup, fastest output
#   fastin : CCLK delay 0.5 (TUSRCCLKO min), 6.5 ns output path, 0.6 ns input path       -> worst output setup, input hold
BASE_VARIANTS = []
for ab in (3, 4):
    for half in (1, 2, 3, 4):
        for late in (1, 0):
            BASE_VARIANTS.append(dict(name=f'a{ab}_h{half}_{"late" if late else "early"}', ab=ab, half=half, late=late))
BASE_VARIANTS += [
    dict(name='pad_h1_late_slowin', ab=3, half=1, late=1, **PADS_SLOWIN),     # the board configuration (HALF=1, SAMPLE_LATE=1, CAP_DLY=2)
    dict(name='pad_h1_late_fastin', ab=3, half=1, late=1, pads=1, sck_d=0.5, out_d=6.5, in_d=0.6, tv=8.0),
    dict(name='pad_h1_late_nodelay', ab=3, half=1, late=1, pads=1, sck_d=0.0, out_d=0.0, in_d=0.0, tv=2.5),
    dict(name='pad_a4_h1_late_slowin', ab=4, half=1, late=1, **PADS_SLOWIN),
    dict(name='pad_h2_late_slowin', ab=3, half=2, late=1, **PADS_SLOWIN),
    dict(name='pad_h3_late_slowin', ab=3, half=3, late=1, **PADS_SLOWIN),
    dict(name='pad_h4_late_slowin', ab=3, half=4, late=1, **PADS_SLOWIN),
    dict(name='pad_a4_h4_late_slowin', ab=4, half=4, late=1, **PADS_SLOWIN),
    # F08: latency code 11 (0 dummy clocks) in the model with DUMMY=0 in the reader; STARTUP_PULSES=3 (the minimum the reader accepts)
    dict(name='lc3_dummy0_a3_h1_late', ab=3, half=1, late=1, lc=3, dummy=0),
    dict(name='pad_lc3_dummy0_h1_late_slowin', ab=3, half=1, late=1, lc=3, dummy=0, **PADS_SLOWIN),
    dict(name='su3_a3_h1_late', ab=3, half=1, late=1, startup=3),
    # N02, other reading of UG470 ("three clock edges" rather than "three clock cycles"): the shim swallows 3 USRCCLKO toggles
    dict(name='sw3edges_a3_h1_late', ab=3, half=1, late=1, swallow=3),
    dict(name='sw3edges_su3_a3_h1_late', ab=3, half=1, late=1, swallow=3, startup=3),
]
VARIANTS = []
for v in BASE_VARIANTS:
    VARIANTS.append(v)
    VARIANTS.append(dict(v, name=v['name'] + '_sp', sparse=1))

# Disagreements between model and reader that must be detected (rc != 0 or mismatches); each is the same configuration as a
# passing variant above with one model-side or shim-side change.
NEGATIVES = [
    dict(name='neg_model_lc3_reader_dummy8', ab=3, half=1, late=1, lc=3, dummy=8, why='model latency code 11 = 0 dummy clocks, reader clocks 8'),
    dict(name='neg_model_lc0_reader_dummy0', ab=3, half=1, late=1, lc=0, dummy=0, why='model latency code 00 = 8 dummy clocks, reader clocks 0'),
    dict(name='neg_model_lc1_reader_dummy0', ab=3, half=1, late=1, lc=1, dummy=0, why='model latency code 01 = 8 dummy clocks, reader clocks 0'),
    dict(name='neg_model_extadd_reader_ab3', ab=3, half=1, late=1, extadd=1, why='model BAR.EXTADD=1 expects 4 address bytes after 6Bh, reader sends 3'),
    dict(name='neg_model_extadd_reader_ab3_sp', ab=3, half=1, late=1, extadd=1, sparse=1, why='same, sparse model'),
    dict(name='neg_startup3_swallow8', ab=3, half=1, late=1, startup=3, swallow=8, why='shim swallows 4 cycles, reader only clocks 3 startup pulses'),
    dict(name='neg_startup3_swallow7', ab=3, half=1, late=1, startup=3, swallow=7, why='shim swallows 7 toggles, reader only clocks 6'),
]
# Parameter values qspi_reader must refuse to elaborate (untested): (reader parameter text, expected undefined-module marker)
REJECTED = [('.HALF(5)', 'HALF'), ('.HALF(0)', 'HALF'), ('.DUMMY(7)', 'DUMMY'), ('.DUMMY(4)', 'DUMMY'), ('.ADDR_BYTES(2)', 'ADDR_BYTES'),
            ('.ADDR_BYTES(5)', 'ADDR_BYTES'), ('.CAP_DLY(1)', 'CAP_DLY'), ('.CAP_DLY(3)', 'CAP_DLY'), ('.STARTUP_PULSES(2)', 'STARTUP_PULSES'),
            ('.STARTUP_PULSES(0)', 'STARTUP_PULSES'), ('.STARTUP_PULSES(5)', 'STARTUP_PULSES'), ('.CS_HIGH(1)', 'CS_HIGH'), ('.CS_HIGH(3)', 'CS_HIGH')]

TB = r"""`timescale 1ns/1ps
// STARTUPE2 CCLK behavior (audit N02). UG470 (7 Series FPGAs Configuration User Guide), STARTUPE2 / USRCCLKO: after the end
// of configuration the CCLK pin does not follow USRCCLKO for the first three clock cycles. The guide could not be re-read
// for this change (PDF host pages unavailable), so the wording "cycles" vs "edges" is treated as ambiguous: SWALLOW counts
// USRCCLKO toggles ignored before the pin follows the input; the default 6 (= 3 full cycles, the worse case for a short startup
// train) is used by every run, and runs named sw3edges_* use 3 (= "three edges"). The pin is held low while swallowing.
module startupe2_cclk_shim #(parameter int SWALLOW = 6) (input wire usrcclko, output wire cclk);
  int cnt = 0; logic prev = 1'bx;
  always @(usrcclko) begin
    if (prev !== 1'bx && usrcclko !== 1'bx && usrcclko !== prev) cnt++;
    prev = usrcclko;
  end
  assign cclk = (cnt > SWALLOW) ? usrcclko : 1'b0;
endmodule

module qspi_tb;
  localparam int NB = @@NB@@, AB = @@AB@@;
  localparam bit SPARSE = @@SPARSE@@;
  localparam logic [31:0] AMASK = (AB == 4) ? 32'hFFFF_FFFF : 32'h00FF_FFFF;
  logic clk = 0, rst = 1; always #10 clk = ~clk;
  logic seek_req = 0, st_ready = 0; logic [31:0] seek_addr = 0;
  wire st_valid; wire [7:0] st_data; wire sclk, cs_n; wire [3:0] io_o, io_oe, f_o, f_oe; tri [3:0] bus;
  wire [3:0] io_i_c;
  wire sck_m, cs_m; wire [3:0] host_o_m, host_oe_m;
  // flash-side wiring. PADS=1: qspi_pad_regs between reader and pads, with board/trace delays in front of the flash model
  // (SCK_D: USRCCLKO->CCLK, OUT_D: pad register -> flash, IN_D: flash -> input register) and CAP_DLY=2.
  // The STARTUPE2 shim sits between the (registered) USRCCLKO and the CCLK pin in both cases.
  localparam bit PADS = @@PADS@@;
  localparam realtime SCK_D = @@SCK_D@@, OUT_D = @@OUT_D@@, IN_D = @@IN_D@@;
  if (PADS) begin : g_pads
    wire pad_sclk, pad_cs_n; wire [3:0] pad_io_o, pad_io_t, bus_in; wire cclk_w;
    qspi_pad_regs pr (.clk, .sclk, .cs_n, .io_o, .io_oe, .pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i(bus_in), .io_i(io_i_c));
    startupe2_cclk_shim #(.SWALLOW(@@SWALLOW@@)) su (.usrcclko(pad_sclk), .cclk(cclk_w));
    assign #(SCK_D) sck_m = cclk_w;
    assign #(OUT_D) cs_m = pad_cs_n;
    assign #(OUT_D) host_o_m = pad_io_o;
    assign #(OUT_D) host_oe_m = ~pad_io_t;
    assign #(IN_D) bus_in = bus;
  end else begin : g_direct
    startupe2_cclk_shim #(.SWALLOW(@@SWALLOW@@)) su (.usrcclko(sclk), .cclk(sck_m));
    assign cs_m = cs_n; assign host_o_m = io_o; assign host_oe_m = io_oe; assign io_i_c = bus;
  end
  for (genvar i = 0; i < 4; i++) begin : g_bus
    assign bus[i] = host_oe_m[i] ? host_o_m[i] : 1'bz;
    assign bus[i] = f_oe[i] ? f_o[i] : 1'bz;
  end
  qspi_reader #(.HALF(@@HALF@@), .SAMPLE_LATE(@@LATE@@)@@XPARAMS@@) dut (.clk, .rst, .seek_req, .seek_addr,
      .st_ready, .st_valid, .st_data, .sclk, .cs_n, .io_o, .io_oe, .io_i(io_i_c));
  qspi_flash_model #(.IMG("@@IMG@@"), .SIZE(NB), .SPARSE(SPARSE), .LC_INIT(@@LC@@), .EXTADD_INIT(@@EXTADD@@), .TV(@@TV@@), .THO(2.0), .TCSS(3.0), .TCSH(3.0), .TSU(3.0), .THD(2.0))
      flash (.sck(sck_m), .cs_n(cs_m), .host_o(host_o_m), .host_oe(host_oe_m), .flash_o(f_o), .flash_oe(f_oe));
  logic [7:0] img [NB]; initial $readmemh("@@IMG@@", img);
  // expected byte: the image, or (sparse) the address hash, written here independently of the model's sparse_byte
  function automatic logic [7:0] exp_byte(input logic [31:0] a);
    longint unsigned h;
    if (SPARSE) begin
      h = longint'(a) ^ (longint'(a) >> 13);
      h = (h * 64'd2246822507) & 64'hFFFF_FFFF;
      h = h ^ (h >> 16);
      return 8'(h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24));
    end else return img[a];
  endfunction
  integer errors = 0, cycles = 0, got = 0, seeks = 0, same_cycle = 0, stall_edges = 0, shown = 0, exp_mut = 0, unexpected_valid = 0;
  logic [31:0] exp_ptr = 0; logic [7:0] eb;
  bit stall_watch = 0;
  logic [31:0] rng = 32'h1234_5678;
  function automatic int rnd(input int n); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return int'(rng % 32'(n)); endfunction
  function automatic logic [31:0] ra();   // random seek address: inside the image, or anywhere in the address space (sparse)
    if (SPARSE) begin rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng & AMASK; end
    else return 32'(rnd(NB - 40));
  endfunction
  always @(posedge clk) begin
    cycles++;
    if (seek_req && st_valid) begin errors++; $display("MISMATCH st_valid high in seek cycle"); end
    if (seek_req) begin seeks++; exp_ptr = seek_addr; if (dut.ob_valid && st_ready) same_cycle++; end
    else if (st_valid && st_ready) begin
      if (!SPARSE && exp_ptr >= NB) begin errors++; unexpected_valid++; if (shown < 8) begin shown++; $display("MISMATCH byte delivered beyond the image at %0d", exp_ptr); end end
      else begin
        eb = exp_byte(exp_ptr);
        if (eb[7:4] != eb[3:0]) exp_mut++;
        if (st_data !== eb) begin errors++; if (shown < 8) begin shown++; $display("MISMATCH addr=%0h got=%h exp=%h", exp_ptr, st_data, eb); end end
      end
      exp_ptr = (exp_ptr + 1) & AMASK; got++;
    end
  end
  task automatic summary();
    $display("SUMMARY errors=%0d cycles=%0d bytes=%0d seeks=%0d same_cycle=%0d stall_edges=%0d exp_mut=%0d over_read=%0d cmds=%0d",
             errors, cycles, got, seeks, same_cycle, stall_edges, exp_mut, flash.over_read, flash.ncmds);
  endtask
  always @(posedge clk) if (cycles > 1500000) begin
    errors++; $display("MISMATCH watchdog: test did not finish");
    summary(); $finish;
  end
  always @(posedge sclk) if (stall_watch && !cs_n) stall_edges++;
  task automatic cyc(input int n); repeat (n) @(negedge clk); endtask
  task automatic seek(input logic [31:0] a); @(negedge clk) seek_addr = a; seek_req = 1; @(negedge clk) seek_req = 0; endtask
  task automatic take(input int n, input int pct);   // accept n bytes, ready asserted pct% of cycles
    int target, guard;
    target = got + n; guard = 0;
    while (got < target) begin
      @(negedge clk); st_ready = (pct >= 100) ? 1 : (rnd(100) < pct);
      guard++;
      if (guard > 400000) begin errors++; $display("MISMATCH timeout waiting for bytes"); summary(); $finish; end
    end
    st_ready = 0;
  endtask
  task automatic sp(input logic [31:0] a, input int n); seek(a & AMASK); take(n, 100); endtask
  initial begin
    cyc(3); rst = 0; cyc(100);                                   // startup pulses + idle
    if (SPARSE) begin
      seek(0); take(16, 100);
      for (int k = 0; k < 8 * AB; k++) begin                     // every address bit, set alone / cleared alone / all lower bits set
        sp(32'd1 << k, 6); sp(AMASK & ~(32'd1 << k), 6); sp((32'd1 << k) - 32'd1, 6);
      end
      sp(32'h0028_0000, 40); sp(32'h0028_1234, 40);              // board image base (FLASH_BASE) and inside it
      sp(32'h0000_3FFE, 6); sp(32'h0000_4000, 6);                // 14-bit boundary
      sp(32'h000F_FFFC, 12); sp(32'h003F_FFFC, 12); sp(32'h007F_FFFC, 12); sp(32'h0080_0000, 8);
      sp(32'h00FF_FFFF - 7, 16);                                 // top of the 16 MiB flash, reads run over the top (wraps to 0)
      sp(32'h00FF_FFFF, 4); sp(32'h00FF_FFFE, 4); sp(32'h00FF_0000, 4);
      sp(32'h00FF_FFFC, 12);                                     // crosses the 24-bit boundary (4-byte: continues at 0x01000000)
      if (AB == 4) begin
        sp(32'h0100_0000, 8); sp(32'h1000_0000, 8); sp(32'h8000_0000, 8); sp(32'hDEAD_BEE0, 16); sp(32'hFFFF_FFF0, 8);
        sp(32'h0128_0000, 8); sp(32'hFF28_0000, 8);
      end
      for (int k = 0; k < 150; k++) begin sp(ra(), 1 + rnd(12)); cyc(rnd(30)); end
      for (int k = 0; k < 40; k++) begin                         // consumer with random backpressure
        seek(ra()); take(4 + rnd(10), 100 - 20 * rnd(4));
      end
      sp(ra(), 5);                                               // long stall: SCLK must stop
      cyc(50); stall_watch = 1; cyc(4000); stall_watch = 0;
      take(100, 100);
      for (int k = 0; k < 60; k++) begin                         // seek in the cycle a byte would be handed off
        sp(ra(), 3 + rnd(5)); st_ready = 1;
        while (!(dut.ob_valid && st_ready)) @(negedge clk);
        seek_addr = ra(); seek_req = 1; @(negedge clk) seek_req = 0; st_ready = 0;
        take(4, 100);
      end
      cyc(100);
      summary();
      $finish;
    end
    seek(0); take(64, 100);                                      // address 0
    seek(NB - 1); take(1, 100); cyc(400);                        // last byte (reader reads on into erased space)
    seek(NB - 2); take(2, 100); cyc(100);
    for (int a = 0; a < 40; a++) begin seek(a); take(17, 100); end   // every alignment
    for (int k = 0; k < 40; k++) begin seek(rnd(NB - 40)); take(1 + rnd(30), 100); end
    seek(100); take(5, 100);                                     // long stall: SCLK must stop
    cyc(50); stall_watch = 1; cyc(4000); stall_watch = 0;
    take(100, 100);
    seek(5000); take(1, 100); cyc(3000); take(300, 25);          // stall in the middle, then slow consumer
    for (int k = 0; k < 60; k++) begin                           // back-to-back seeks
      int n; n = 1 + rnd(4);
      for (int j = 0; j < n; j++) begin @(negedge clk) seek_addr = rnd(NB - 40); seek_req = 1; @(negedge clk) seek_req = 0; cyc(rnd(3)); end
      take(1 + rnd(20), 100);
    end
    for (int k = 0; k < 250; k++) begin                          // seeks while a burst is running, mid-byte/mid-command
      seek(rnd(NB - 40)); st_ready = 1; cyc(rnd(140) + 1); st_ready = rnd(2);
      if (rnd(3) == 0) take(1 + rnd(8), 50);
    end
    st_ready = 0;
    for (int k = 0; k < 60; k++) begin                           // seek in the cycle a byte would be handed off
      seek(rnd(NB - 40)); take(3 + rnd(5), 100); st_ready = 1;
      while (!(dut.ob_valid && st_ready)) @(negedge clk);
      seek_addr = rnd(NB - 40); seek_req = 1; @(negedge clk) seek_req = 0; st_ready = 0;
      take(4, 100);
    end
    for (int k = 0; k < 150; k++) begin seek(rnd(NB)); take(1, 100 - 20 * rnd(4)); cyc(rnd(60)); end
    for (int k = 0; k < 3; k++) begin seek(0); take(4000, 70 + rnd(31)); end   // long streaming reads with a flaky consumer
    cyc(100);
    summary();
    $finish;
  end
endmodule
"""


def build_tb(v, img_path):
    pads = v.get('pads', 0)
    # parameters left at their default are NOT passed (the board's boolean_core instantiates qspi_reader the same way), so a
    # change of a default value (ADDR_BYTES=3, DUMMY=8, CAP_DLY=0, STARTUP_PULSES=4) is visible to the test
    xp = ''
    if v['ab'] != 3: xp += f", .ADDR_BYTES({v['ab']})"
    if 'dummy' in v: xp += f", .DUMMY({v['dummy']})"
    if pads: xp += ', .CAP_DLY(2)'
    if 'startup' in v: xp += f", .STARTUP_PULSES({v['startup']})"
    xp += v.get('xparams', '')
    sub = dict(NB=NB, AB=v['ab'], SPARSE=v.get('sparse', 0), PADS=pads, SCK_D=v.get('sck_d', 0.0), OUT_D=v.get('out_d', 0.0),
               IN_D=v.get('in_d', 0.0), SWALLOW=v.get('swallow', 6), HALF=v['half'], LATE=v['late'], XPARAMS=xp, IMG=img_path, LC=v.get('lc', 0), EXTADD=v.get('extadd', 0), TV=v.get('tv', 7.65))
    t = TB
    for k, x in sub.items():
        t = t.replace(f'@@{k}@@', str(x))
    assert '@@' not in t
    return t


def sim(srcs, tb, defines=()):
    """compile+run; a compile/elaboration failure is returned, not raised (it counts as a detection for mutants)."""
    try:
        rc, out = compile_and_run(srcs, 'qspi_tb', tb, defines=defines)
        return rc, out, ''
    except subprocess.CalledProcessError as e:
        return -1, '', (e.stderr or '') + (e.stdout or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', default=str(ROOT / 'rtl/decode/qspi_reader.sv'))
    ap.add_argument('--output', default=str(ROOT / 'bench/results/decode_qspi_rtl.json'))
    ap.add_argument('--variants', default=None, help='comma list of variant names (default all)')
    a = ap.parse_args()
    want = set(a.variants.split(',')) if a.variants else None
    img = np.random.default_rng(20261027).integers(0, 256, NB, dtype=np.uint8)
    img_path = write_mem('qspi_img', list(img), 8)
    model = ROOT / 'tools/decode/tb/qspi_flash_model.sv'
    srcs = [a.rtl, model, ROOT / 'rtl/board/qspi_pad_regs.sv']
    BUILD.mkdir(parents=True, exist_ok=True)
    runs, tot_err, tot_mut, tot_exp = [], 0, 0, 0
    for v in VARIANTS:
        name = v['name']
        if want and name not in want:
            continue
        pads, sparse = v.get('pads', 0), v.get('sparse', 0)
        tb = BUILD / f'qspi_{name}_tb.sv'
        tb.write_text(build_tb(v, img_path))
        rc, out, cerr = sim(srcs, tb)
        mrc, mout, mcerr = sim(srcs, tb, ('QSPI_MUTATE',))
        line, kv = parse_summary(out)
        mline, mkv = parse_summary(mout)
        r = {'variant': name, 'mode': 'sparse' if sparse else 'image', 'addr_bytes': v['ab'], 'half': v['half'], 'sample_late': v['late'],
             'pad_regs': pads, 'cap_dly': 2 if pads else 0, 'dummy': v.get('dummy', 8), 'flash_lc': v.get('lc', 0),
             'startup_pulses': v.get('startup', 'default'), 'startupe2_swallow_toggles': v.get('swallow', 6),
             'delays_ns': {k: v[k] for k in ('sck_d', 'out_d', 'in_d', 'tv') if k in v}, 'rc': rc, 'summary': line, 'mutation_summary': mline,
             'mismatches': (int(kv['errors']) if 'errors' in kv else 0) + (1 if rc != 0 or 'errors' not in kv else 0),  # a model $fatal or an elaboration error counts as a detection
             'bytes_checked': int(kv.get('bytes', 0)), 'seeks': int(kv.get('seeks', 0)),
             'same_cycle_seek_handoffs': int(kv.get('same_cycle', 0)), 'sclk_edges_during_4000_cycle_stall': int(kv.get('stall_edges', -1)),
             'flash_commands': int(kv.get('cmds', 0)), 'reads_past_image_end_by_reader_lookahead': int(kv.get('over_read', 0)),
             'mutation_mismatches': int(mkv.get('errors', -1)), 'mutation_expected': int(mkv.get('exp_mut', -2)),
             'fatal_lines': [x for x in (out + mout).splitlines() if 'FLASH' in x or 'MISMATCH' in x][:4] + ([cerr.strip().splitlines()[0]] if cerr.strip() else []),
             'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest()}
        r['ok'] = (rc == 0 and r['mismatches'] == 0 and r['bytes_checked'] > (1000 if sparse else 10000) and 0 <= r['sclk_edges_during_4000_cycle_stall'] <= 4
                   and r['mutation_mismatches'] == r['mutation_expected'] > 0 and r['same_cycle_seek_handoffs'] > 0)
        runs.append(r)
        tot_err += r['mismatches']; tot_mut += r['mutation_mismatches']; tot_exp += r['mutation_expected']
        print(name, r['ok'], line, '| mut', mline, r['fatal_lines'], flush=True)

    # negative tests: model/reader/shim disagreements that MUST be detected (mismatch, model $fatal, or watchdog), never a crash or a pass
    negs = []
    for v in NEGATIVES:
        name = v['name']
        if want and name not in want:
            continue
        tb = BUILD / f'qspi_{name}_tb.sv'
        tb.write_text(build_tb(v, img_path))
        rc, out, cerr = sim(srcs, tb)
        line, kv = parse_summary(out)
        det_mism = int(kv['errors']) if 'errors' in kv else 0
        fatals = [x for x in out.splitlines() if 'FLASH' in x or 'MISMATCH' in x]
        detected = (cerr == '') and (rc != 0 or det_mism > 0 or 'errors' not in kv)
        negs.append({'variant': name, 'why': v['why'], 'rc': rc, 'summary': line, 'mismatches': det_mism, 'fatal_lines': fatals[:3],
                     'detected': detected, 'ok': detected})
        print('NEG', name, detected, line, fatals[:2], flush=True)

    # untested parameter values are rejected at elaboration
    rej = []
    for xp, par in REJECTED:
        if want and 'rejected' not in want:
            continue
        v = dict(ab=3, half=1, late=1, xparams=', ' + xp)
        if xp.startswith(('.HALF', '.DUMMY', '.ADDR_BYTES')):   # these are set by build_tb from the variant: override them there
            key = {'.HALF': 'half', '.DUMMY': 'dummy', '.ADDR_BYTES': 'ab'}[xp.split('(')[0]]
            v = dict(ab=3, half=1, late=1)
            v[key] = int(xp.split('(')[1].rstrip(')'))
        tb = BUILD / f'qspi_rej_{par}_{xp.split("(")[1].rstrip(")")}_tb.sv'
        text = build_tb(v, img_path)
        tb.write_text(text)
        rc, out, cerr = sim(srcs, tb)
        marker = f'qspi_reader_untested_parameter_{par}'
        good = rc == -1 and marker in cerr
        rej.append({'param': xp, 'rejected': good, 'ok': good})
        print('REJECT', xp, good, flush=True)

    ok = bool(runs) and all(r['ok'] for r in runs) and all(n['ok'] for n in negs) and all(x['ok'] for x in rej)
    o = {'status': 'pass' if ok else 'fail', 'image_bytes': NB, 'image_seed': 20261027,
         'flash_model': 'tools/decode/tb/qspi_flash_model.sv', 'vvp_command': f'{VVP} build/decode/qspi_*_tb.vvp',
         'mismatch_count': tot_err, 'real_mismatch_count': tot_err, 'mutation_mismatch_count': tot_mut, 'mutation_expected': tot_exp, 'mutation': 'QSPI_MUTATE (nibble swap)',
         'bytes_checked': sum(r['bytes_checked'] for r in runs), 'variants': runs,
         'negative_tests': negs, 'negative_tests_detected': sum(n['detected'] for n in negs),
         'rejected_parameters': rej,
         'provenance': provenance(ARTIFACT_SOURCES['decode_qspi_rtl'], {'rtl/decode/qspi_reader.sv': a.rtl})}
    finish(o, a.output, ('status',))


if __name__ == '__main__':
    main()
