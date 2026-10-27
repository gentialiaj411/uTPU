"""Unit test: rtl/decode/qspi_reader.sv against the independent flash model (tools/decode/tb/qspi_flash_model.sv).

Directed + randomized cases (seek to 0 / last byte / every alignment, back-to-back seeks, seeks mid-byte,
long backpressure stalls, a seek in the same cycle as a byte handoff) over several reader configurations.
Every delivered byte is compared with the image. A -DQSPI_MUTATE build (nibble swap) must be caught on
exactly the bytes whose nibbles differ.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from harness import ROOT, BUILD, write_mem, parse_summary
from sim import compile_and_run, VVP

NB = 16384
# (name, ADDR_BYTES, HALF, SAMPLE_LATE)
# direct pads (no pad registers); then behind qspi_pad_regs with CAP_DLY=2 and board/flash delay corners:
#   slowin : CCLK delay 7.5 (DS189 TUSRCCLKO max), flash tV 8.0 (max), 2.0 ns input path  -> worst input setup, fastest output
#   fastin : CCLK delay 0.5 (TUSRCCLKO min), 6.5 ns output path, 0.6 ns input path       -> worst output setup, input hold
VARIANTS = [dict(name='a3_h1_late', ab=3, half=1, late=1), dict(name='a3_h1_early', ab=3, half=1, late=0),
            dict(name='a3_h2_late', ab=3, half=2, late=1), dict(name='a4_h1_late', ab=4, half=1, late=1),
            dict(name='a3_h3_early', ab=3, half=3, late=0),
            dict(name='pad_h1_late_slowin', ab=3, half=1, late=1, pads=1, sck_d=7.5, out_d=4.0, in_d=2.0, tv=8.0),
            dict(name='pad_h1_late_fastin', ab=3, half=1, late=1, pads=1, sck_d=0.5, out_d=6.5, in_d=0.6, tv=8.0),
            dict(name='pad_h1_late_nodelay', ab=3, half=1, late=1, pads=1, sck_d=0.0, out_d=0.0, in_d=0.0, tv=2.5),
            dict(name='pad_a4_h1_late_slowin', ab=4, half=1, late=1, pads=1, sck_d=7.5, out_d=4.0, in_d=2.0, tv=8.0),
            dict(name='pad_h2_late_slowin', ab=3, half=2, late=1, pads=1, sck_d=7.5, out_d=4.0, in_d=2.0, tv=8.0)]

TB = r"""`timescale 1ns/1ps
module qspi_tb;
  localparam int NB = %(nb)d;
  logic clk = 0, rst = 1; always #10 clk = ~clk;
  logic seek_req = 0, st_ready = 0; logic [31:0] seek_addr = 0;
  wire st_valid; wire [7:0] st_data; wire sclk, cs_n; wire [3:0] io_o, io_oe, f_o, f_oe; tri [3:0] bus;
  wire [3:0] io_i_c;
  wire sck_m, cs_m; wire [3:0] host_o_m, host_oe_m;
  // flash-side wiring. PADS=1: qspi_pad_regs between reader and pads, with board/trace delays in front of the flash model
  // (SCK_D: USRCCLKO->CCLK, OUT_D: pad register -> flash, IN_D: flash -> input register) and CAP_DLY=2.
  localparam bit PADS = %(pads)d;
  localparam realtime SCK_D = %(sck_d)s, OUT_D = %(out_d)s, IN_D = %(in_d)s;
  if (PADS) begin : g_pads
    wire pad_sclk, pad_cs_n; wire [3:0] pad_io_o, pad_io_t, bus_in;
    qspi_pad_regs pr (.clk, .sclk, .cs_n, .io_o, .io_oe, .pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i(bus_in), .io_i(io_i_c));
    assign #(SCK_D) sck_m = pad_sclk;
    assign #(OUT_D) cs_m = pad_cs_n;
    assign #(OUT_D) host_o_m = pad_io_o;
    assign #(OUT_D) host_oe_m = ~pad_io_t;
    assign #(IN_D) bus_in = bus;
  end else begin : g_direct
    assign sck_m = sclk; assign cs_m = cs_n; assign host_o_m = io_o; assign host_oe_m = io_oe; assign io_i_c = bus;
  end
  for (genvar i = 0; i < 4; i++) begin : g_bus
    assign bus[i] = host_oe_m[i] ? host_o_m[i] : 1'bz;
    assign bus[i] = f_oe[i] ? f_o[i] : 1'bz;
  end
  qspi_reader #(.ADDR_BYTES(%(ab)d), .HALF(%(half)d), .SAMPLE_LATE(%(late)d), .DUMMY(8), .CAP_DLY(%(cap)d)) dut (.clk, .rst, .seek_req, .seek_addr,
      .st_ready, .st_valid, .st_data, .sclk, .cs_n, .io_o, .io_oe, .io_i(io_i_c));
  qspi_flash_model #(.IMG("%(img)s"), .SIZE(NB), .ADDR_BYTES(%(ab)d), .DUMMY(8), .TV(%(tv)s), .THO(2.0), .TCSS(3.0), .TCSH(3.0), .TSU(3.0), .THD(2.0))
      flash (.sck(sck_m), .cs_n(cs_m), .host_o(host_o_m), .host_oe(host_oe_m), .flash_o(f_o), .flash_oe(f_oe));
  logic [7:0] img [NB]; initial $readmemh("%(img)s", img);
  integer errors = 0, cycles = 0, got = 0, seeks = 0, same_cycle = 0, stall_edges = 0, shown = 0, exp_mut = 0, unexpected_valid = 0;
  integer exp_ptr = 0;
  bit stall_watch = 0;
  logic [31:0] rng = 32'h1234_5678;
  function automatic int rnd(input int n); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return int'(rng %% 32'(n)); endfunction
  always @(posedge clk) begin
    cycles++;
    if (seek_req && st_valid) begin errors++; $display("MISMATCH st_valid high in seek cycle"); end
    if (seek_req) begin seeks++; exp_ptr = seek_addr; if (dut.ob_valid && st_ready) same_cycle++; end
    else if (st_valid && st_ready) begin
      if (exp_ptr >= NB) begin errors++; unexpected_valid++; if (shown < 8) begin shown++; $display("MISMATCH byte delivered beyond the image at %%0d", exp_ptr); end end
      else begin
        if (img[exp_ptr][7:4] != img[exp_ptr][3:0]) exp_mut++;
        if (st_data !== img[exp_ptr]) begin errors++; if (shown < 8) begin shown++; $display("MISMATCH addr=%%0d got=%%h exp=%%h", exp_ptr, st_data, img[exp_ptr]); end end
      end
      exp_ptr++; got++;
    end
  end
  always @(posedge clk) if (cycles > 1500000) begin
    errors++; $display("MISMATCH watchdog: test did not finish");
    $display("SUMMARY errors=%%0d cycles=%%0d bytes=%%0d seeks=%%0d same_cycle=%%0d stall_edges=%%0d exp_mut=%%0d over_read=%%0d cmds=%%0d", errors, cycles, got, seeks, same_cycle, stall_edges, exp_mut, flash.over_read, flash.ncmds);
    $finish;
  end
  always @(posedge sclk) if (stall_watch && !cs_n) stall_edges++;
  task automatic cyc(input int n); repeat (n) @(negedge clk); endtask
  task automatic seek(input int a); @(negedge clk) seek_addr = a; seek_req = 1; @(negedge clk) seek_req = 0; endtask
  task automatic take(input int n, input int pct);   // accept n bytes, ready asserted pct%% of cycles
    int target, guard;
    target = got + n; guard = 0;
    while (got < target) begin
      @(negedge clk); st_ready = (pct >= 100) ? 1 : (rnd(100) < pct);
      guard++;
      if (guard > 400000) begin errors++; $display("MISMATCH timeout waiting for bytes"); $display("SUMMARY errors=%%0d cycles=%%0d bytes=%%0d seeks=%%0d same_cycle=%%0d stall_edges=%%0d exp_mut=%%0d over_read=%%0d cmds=%%0d", errors, cycles, got, seeks, same_cycle, stall_edges, exp_mut, flash.over_read, flash.ncmds); $finish; end
    end
    st_ready = 0;
  endtask
  initial begin
    cyc(3); rst = 0; cyc(100);                                   // startup pulses + idle
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
    $display("SUMMARY errors=%%0d cycles=%%0d bytes=%%0d seeks=%%0d same_cycle=%%0d stall_edges=%%0d exp_mut=%%0d over_read=%%0d cmds=%%0d",
             errors, cycles, got, seeks, same_cycle, stall_edges, exp_mut, flash.over_read, flash.ncmds);
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtl', default=str(ROOT / 'rtl/decode/qspi_reader.sv'))
    ap.add_argument('--output', default=str(ROOT / 'bench/results/decode_qspi_rtl.json'))
    ap.add_argument('--variants', default=None, help='comma list of variant names (default all)')
    a = ap.parse_args()
    img = np.random.default_rng(20261027).integers(0, 256, NB, dtype=np.uint8)
    img_path = write_mem('qspi_img', list(img), 8)
    model = ROOT / 'tools/decode/tb/qspi_flash_model.sv'
    runs, tot_err, tot_mut, tot_exp = [], 0, 0, 0
    for v in VARIANTS:
        name, ab, half, late = v['name'], v['ab'], v['half'], v['late']
        if a.variants and name not in a.variants.split(','):
            continue
        pads = v.get('pads', 0)
        tb = BUILD / f'qspi_{name}_tb.sv'
        tb.write_text(TB % dict(nb=NB, ab=ab, half=half, late=late, img=img_path, pads=pads, cap=2 if pads else 0,
                                sck_d=v.get('sck_d', 0.0), out_d=v.get('out_d', 0.0), in_d=v.get('in_d', 0.0), tv=v.get('tv', 7.65)))
        rc, out = compile_and_run([a.rtl, model, ROOT / 'rtl/board/qspi_pad_regs.sv'], 'qspi_tb', tb)
        mrc, mout = compile_and_run([a.rtl, model, ROOT / 'rtl/board/qspi_pad_regs.sv'], 'qspi_tb', tb, defines=('QSPI_MUTATE',))
        line, kv = parse_summary(out)
        mline, mkv = parse_summary(mout)
        r = {'variant': name, 'addr_bytes': ab, 'half': half, 'sample_late': late, 'pad_regs': pads, 'cap_dly': 2 if pads else 0, 'delays_ns': {k: v[k] for k in ('sck_d', 'out_d', 'in_d', 'tv') if k in v}, 'rc': rc, 'summary': line, 'mutation_summary': mline,
             'mismatches': (int(kv['errors']) if 'errors' in kv else 0) + (1 if rc != 0 or 'errors' not in kv else 0),  # a model $fatal counts as a detection
             'bytes_checked': int(kv.get('bytes', 0)), 'seeks': int(kv.get('seeks', 0)),
             'same_cycle_seek_handoffs': int(kv.get('same_cycle', 0)), 'sclk_edges_during_4000_cycle_stall': int(kv.get('stall_edges', -1)),
             'flash_commands': int(kv.get('cmds', 0)), 'reads_past_image_end_by_reader_lookahead': int(kv.get('over_read', 0)),
             'mutation_mismatches': int(mkv.get('errors', -1)), 'mutation_expected': int(mkv.get('exp_mut', -2)),
             'fatal_lines': [x for x in (out + mout).splitlines() if 'FLASH' in x or 'MISMATCH' in x][:4],
             'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest()}
        r['ok'] = (rc == 0 and r['mismatches'] == 0 and r['bytes_checked'] > 10000 and 0 <= r['sclk_edges_during_4000_cycle_stall'] <= 4
                   and r['mutation_mismatches'] == r['mutation_expected'] > 0 and r['same_cycle_seek_handoffs'] > 0)
        runs.append(r)
        tot_err += r['mismatches']; tot_mut += r['mutation_mismatches']; tot_exp += r['mutation_expected']
        print(name, r['ok'], line, '| mut', mline, r['fatal_lines'])
    o = {'status': 'pass' if runs and all(r['ok'] for r in runs) else 'fail', 'image_bytes': NB, 'image_seed': 20261027,
         'flash_model': 'tools/decode/tb/qspi_flash_model.sv', 'vvp_command': f'{VVP} build/decode/qspi_*_tb.vvp',
         'mismatch_count': tot_err, 'real_mismatch_count': tot_err, 'mutation_mismatch_count': tot_mut, 'mutation_expected': tot_exp, 'mutation': 'QSPI_MUTATE (nibble swap)',
         'bytes_checked': sum(r['bytes_checked'] for r in runs), 'variants': runs}
    Path(a.output).write_text(json.dumps(o, indent=2) + '\n')
    print(o['status'])


if __name__ == '__main__':
    main()
