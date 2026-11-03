"""Fault-inject decode RTL by source-text edits and confirm each runner reports mismatches.

Usage: python tools/decode/fault_inject.py [--out FILE] <unit> [<unit> ...]
Results merge into bench/results/decode_fault_injection.json (or --out FILE).
A runner exits 1 when its artifact says status=fail; that is a normal 'mutant detected' outcome only if the artifact was
written with status 'fail' and real (non-sentinel) mismatch counts. A crash, compile error, timeout or missing artifact is
recorded as an error (ok=False), never as a detection.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import provenance, FAULT_SOURCES
ROOT = Path(__file__).resolve().parents[2]
SENTINEL = 10 ** 9      # runners report 1e9 mismatches when the simulation itself failed (rc != 0 / no SUMMARY)
RUN_TIMEOUT_S = 6 * 3600
PKG = 'rtl/decode/decode_pkg.sv'

# unit -> (runner, {fault: (file, old, new)}); 'file' is the RTL or the shared package.
FAULTS = {
    'rms': ('tools/decode/run_rms_rtl.py', 'rtl/decode/rmsnorm.sv', {
        'mean_no_correction': ('rtl', "qf <= div_const_fix_resid(ss_mag > 64'h8000_0000, q0, r0, 10'(count));", "qf <= q0;"),
        'small_path_half_up': ('pkg', "(frac == 32'h8000_0000 && q[0])", "(frac == 32'h8000_0000)",
                               'equivalent: on an exact tie the half-up estimate is odd and the tie_odd correction restores RNE'),
        'recip_half_up': ('rtl', "== {1'b0, 56'(root_eff)}) && dv_ge)", "== {1'b0, 56'(root_eff)}))",
                          'equivalent: 2^54 mod root == root/2 has no solution, so div_rne(2^54, root) never ties'),
        'gamma_shift_off_by_one': ('rtl', "(step == 2'd1) ? 7'(gfb)", "(step == 2'd1) ? 7'(gfb) + 7'd1"),
        'seq_mul_drop_top_nibble': ('rtl', "mul_b <= 64'(recip); mul_start", "mul_b <= 64'(recip[33:0]); mul_start"),
        'shift_ties_away': ('pkg', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'isqrt_no_max1': ('rtl', "wire [46:0] root_eff = (root == 0) ? 47'd1 : root;", "wire [46:0] root_eff = root | 47'd1;"),
    }),
    'vec': ('tools/decode/run_vec_rtl.py', 'rtl/decode/vec_alu.sv', {
        'rope_sign': ('rtl', "r0 = 32'(rshift_rne(t0 - t1, 7'd15));", "r0 = 32'(rshift_rne(t0 + t1, 7'd15));"),
        'rope_h48_only_bitflip': ('rtl', "r0 = 32'(rshift_rne(t0 - t1, 7'd15));", "r0 = 32'(rshift_rne(t0 - t1, 7'd15)) ^ ((rope_half == 6'd24) ? 32'd2 : 32'd0);"),   # N07: only the 15M (HS=48) geometry corrupts
        'rope_cos_sin_lane_swap': ('rtl', "{c_r, s_r} <= rope_rom", "{s_r, c_r} <= rope_rom"),
        'rope_no_pair_wrap': ('rtl', "(pair_i + 1'b1 == rope_half) ? '0 :", "1'b0 ? '0 :"),
        'sigmoid_edge': ('rtl', "(idx_r > 64'sd512) ? 16'd32767", "(idx_r >= 64'sd512) ? 16'd32767"),
        'swiglu_drop_high_word': ('rtl', "wire signed [63:0] low64 = t2 + (t3 <<< 32);", "wire signed [63:0] low64 = t2;"),
        'shift_ties_away': ('pkg', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'add_saturate': ('rtl', "OP_ADD: r0 = a_r + b_r;", "OP_ADD: r0 = (33'(a_r) + 33'(b_r) > 33'sd2147483647) ? 32'sd2147483647 : a_r + b_r;"),
    }),
    'attn': ('tools/decode/run_attn_rtl.py', 'rtl/decode/attention.sv', {
        'exp_clip_1023': ('rtl', "(d > 34'sd1024) ? 11'd1024 : 11'(d)", "(d > 34'sd1023) ? 11'd1023 : 11'(d)",
                          'equivalent: exp_q15 is 0 for every index >= 710, so clipping at 1023 or 1024 reads 0'),
        'exp_clip_512': ('rtl', "(d > 34'sd1024) ? 11'd1024 : 11'(d)", "(d > 34'sd512) ? 11'd512 : 11'(d)"),
        'kv4_score_div127': ('rtl', "dp_d127 <= !kv4; state <= SC_DWAIT;", "dp_d127 <= 1'b1; state <= SC_DWAIT;"),
        'av_clip_127': ('rtl', "(r < -64'sd128) ? -32'sd128", "(r < -64'sd127) ? -32'sd127"),
        'softmax_recip_trunc': ('rtl', "precip <= dv_qn + 40'(dv_up);", "precip <= dv_qn;"),
        'kv_v_region_offset': ('rtl', "AW'(32'(blk + TW'(i)) * KV_HEADS)) + DEPTH);", "AW'(32'(blk + TW'(i)) * KV_HEADS)) + 0);"),
        'kv_tail_bank_half': ('rtl', "localparam int DB  = (bk < NF) ? BD : (1 << $clog2(REM));", "localparam int DB  = (bk < NF) ? BD : (1 << $clog2(REM)) / 2;"),
        'kv_write_v_offset': ('rtl', "+ (kv_is_v ? DEPTH : 0));", "+ (kv_is_v ? DEPTH - 1 : 0));"),
        'kv_bank_sel_undelayed': ('rtl', "<= bq[32'(raddr_d) >> BAW];", "<= bq[32'(raddr) >> BAW];",
                                  'equivalent: raddr is written in SC_RD/AV_VRD and kvq_r is first consumed three states later (SC_DOT/AV_VISSUE), by which time raddr == raddr_d'),
        'kv4_zero_extend': ('rtl', "return 8'($signed(w[j*CB +: CB]));", "return 8'(w[j*CB +: CB]);"),
        'block_recip_trunc': ('rtl', "vrecip <= 32'(dv_qn + 40'(dv_up));", "vrecip <= 32'(dv_qn);"),
    }),
    'actq': ('tools/decode/run_act_quant_rtl.py', 'rtl/decode/act_quant.sv', {
        'ties_half_up': ('rtl', "((rem24 == 24'h800000) && q[0])", "(rem24 == 24'h800000)"),
        'recip_truncate': ('rtl', "quot_next + 32'(({rem_next", "quot_next + 32'(1'b0 && ({rem_next"),
        'peak_off_by_one': ('rtl', "if (mag_in > peak) peak <= mag_in;", "if (mag_in > peak) peak <= mag_in + 32'd1;"),
        'qmax7_clamp_8': ('rtl', "(q7 ? 64'd7 : 64'd127)) q = q7 ? 64'd7", "(q7 ? 64'd8 : 64'd127)) q = q7 ? 64'd8"),
        'qmax7_numerator_127': ('rtl', "qmax7_r ? 31'(7) << 24", "qmax7_r ? 31'(127) << 24"),
    }),
    'top': ('tools/decode/run_decode_rtl.py --prompts 1 --gen 8', 'rtl/decode/top_decode.sv', {
        'missing_effective_multiplier_expansion': ('rtl', "{btmp[31:16], 16'b0}", "{16'b0, btmp[31:16]}"),
        # new in the BRAM move: byte-lane writes into the 128-bit tbuf word, per-row scale files, role-sized scratch
        'tbuf_lane_swap': ('rtl', "assign tb_wr = (state == E_CODES) ? 4'd0 : rr[3:0];", "assign tb_wr = (state == E_CODES) ? 4'd0 : rr[3:0] ^ 4'd1;"),
        'tbuf_embed_addr': ('rtl', "assign tb_wa = (state == E_CODES) ? g_kb : i;", "assign tb_wa = (state == E_CODES) ? g_kb + LW'(1) : i;"),
        'scale_write_group': ('rtl', "assign sc_wg = gg;", "assign sc_wg = 6'd0;"),
        'scale_embed_read_group': ('rtl', "assign sc_ra         = (state == E_ISSUE) ? LW'(i >> 5) : LW'(kk >> 5);", "assign sc_ra         = LW'(kk >> 5);"),
        'scale_row_select': ('rtl', "assign sc_wr = (state == E_SCALES) ? 5'd0 : rr;", "assign sc_wr = (state == E_SCALES) ? 5'd0 : (rr ^ 5'd1);"),
        'scratch_xb_overlap': ('rtl', "BXB:     off = SSZ;", "BXB:     off = SSZ - 1;"),
        'scratch_h3_overlap': ('rtl', "default: off = 6 * SSZ + H;", "default: off = 6 * SSZ + H - 1;"),
        # verification-integrity faults at the logit output (audit F01): the e2e TB must check the emitted index sequence,
        # not just the value at whatever index the DUT claims, and must reach positions >= 9
        'top_duplicate_idx0': ('rtl', "logit_valid <= 1'b1; logit_idx <= row; logit_val <= v;", "logit_valid <= 1'b1; logit_idx <= 16'd0; logit_val <= v;"),
        'top_idx_skip': ('rtl', "logit_valid <= 1'b1; logit_idx <= row; logit_val <= v;", "logit_valid <= 1'b1; logit_idx <= row + 16'd1; logit_val <= v;"),
        'top_late_pos': ('rtl', "logit_valid <= 1'b1; logit_idx <= row; logit_val <= v;", "logit_valid <= 1'b1; logit_idx <= row; logit_val <= (pos >= 9) ? (v ^ 32'sd1) : v;"),
    }),
    'qspi': ('tools/decode/run_qspi_rtl.py', 'rtl/decode/qspi_reader.sv', {
        'dummy_count_off_by_one': ('rtl', "localparam int PRE = SHW + DUMMY; ", "localparam int PRE = SHW + DUMMY - 1; "),
        'nibble_order_swapped': ('rtl', "ob_data <= {nib_hi, nib};", "ob_data <= {nib, nib_hi};"),
        'address_byte_order_swapped': ('rtl', "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0]};", "sh <= {OPC, pend_addr[7:0], pend_addr[15:8], pend_addr[8*ADDR_BYTES-1:16]};"),
        'wrong_opcode_fast_read': ('rtl', "(ADDR_BYTES == 4) ? 8'h6C : 8'h6B;", "(ADDR_BYTES == 4) ? 8'h6C : 8'h0B;"),
        'cs_not_raised_on_seek': ('rtl', "cs_n_q <= 1'b1; io_oe_q <= 4'h0; io_o_q <= 4'hF;", "io_oe_q <= 4'h0; io_o_q <= 4'hF;"),
        'byte_valid_in_seek_cycle': ('rtl', "assign st_valid = ob_valid && !seek_req;", "assign st_valid = ob_valid;"),
        'seek_keeps_stale_byte': ('rtl', "pend_addr <= seek_addr; ob_valid <= 1'b0; end", "pend_addr <= seek_addr; end"),
        'no_backpressure_stall': ('rtl', "wire rise_blocked = !sclk_q && data_phase && eff_ph && !(room && !pend_complete);", "wire rise_blocked = 1'b0;"),
        'inflight_captures_ignored': ('rtl', "wire rise_blocked = !sclk_q && data_phase && eff_ph && !(room && !pend_complete);", "wire rise_blocked = !sclk_q && data_phase && nib_ph && !room;"),
        'capture_strobe_one_cycle_early': ('rtl', "wire strobe      = (CAP_DLY == 0) ? launch : cap_p[CD-1];", "wire strobe      = (CAP_DLY == 0) ? launch : cap_p[(CD > 1) ? CD-2 : 0];"),
        'abort_mid_command_byte': ('rtl', "(data_phase || rise_n[2:0] == 3'd0);", "1'b1;"),
        'stale_capture_after_seek': ('rtl', "if (!seek_req && !pend) begin", "if (1'b1) begin"),
        'cs_high_gap_one_cycle': ('rtl', "state <= S_GAP; gcnt <= 4'(CS_HIGH - 1); cs_n_q <= 1'b1;", "state <= S_GAP; gcnt <= 4'd0; cs_n_q <= 1'b1;",
                                  'equivalent ONLY under the assumption that the core clock is 50 MHz (20 ns period; the unit test uses exactly that, the board runs the core at 50 MHz): CS# then stays high one core cycle (20 ns) which already exceeds tCS(read) = 10 ns, so CS_HIGH=2 is only margin. At a core clock faster than 100 MHz (period < 10 ns) this fault WOULD violate tCS and the model TCS check would catch it; the e2e sim uses a time-scaled 2 ns clock with a scaled tCS of 1 ns and so stays equivalent there too'),
        # audit N01: address path above bit 13 (the unit image alone is 16 KiB); caught by the sparse full-address-space runs
        'seek_address_masked_14_bits': ('rtl', "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0]};", "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0] & {{(8*ADDR_BYTES-14){1'b0}}, 14'h3FFF}};"),
        'seek_address_bit24_dropped_4byte': ('rtl', "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0]};", "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0] & ~((8*ADDR_BYTES)'(1) << 24)};"),
        'seek_address_bit21_dropped': ('rtl', "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0]};", "sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0] & ~((8*ADDR_BYTES)'(1) << 21)};"),
        # audit N02: STARTUPE2 swallows the first USRCCLKO cycles (startupe2_cclk_shim in the unit TB)
        'startup_pulses_not_clocked': ('rtl', "su_cnt <= 8'(2 * STARTUP_PULSES);", "su_cnt <= 8'(0 * STARTUP_PULSES);"),
        'startup_pulses_default_zero': ('rtl', "parameter int STARTUP_PULSES = 4,", "parameter int STARTUP_PULSES = 0,"),
        'startup_pulses_default_two': ('rtl', "parameter int STARTUP_PULSES = 4,", "parameter int STARTUP_PULSES = 2,"),
        # audit N04: corruption that exists only at one HALF value
        'half4_only_nibble_swap': ('rtl', "ob_data <= {nib_hi, nib};", "ob_data <= (HALF == 4) ? {nib, nib_hi} : {nib_hi, nib};"),
        'half3_only_nibble_swap': ('rtl', "ob_data <= {nib_hi, nib};", "ob_data <= (HALF == 3) ? {nib, nib_hi} : {nib_hi, nib};"),
        # audit F08: reader assumes latency code 00 (8 dummy clocks); the model decodes CR1[7:6]
        'dummy_clocks_assume_lc11_zero': ('rtl', "localparam int PRE = SHW + DUMMY; ", "localparam int PRE = SHW + 0; "),
        'dummy_default_untested_value': ('rtl', "parameter int DUMMY          = 8,", "parameter int DUMMY          = 5,"),
        'abort_leaves_nib_ph': ('rtl', "ob_valid <= 1'b0; nib_ph <= 1'b0; tcnt <= '0;", "ob_valid <= 1'b0; tcnt <= '0;",
                                'equivalent: the S_GAP exit that starts the next command also clears nib_ph, and nibbles are only captured in S_RUN, so the stale value is never observed'),
    }),
    'board': ('tools/board/run_uart_rtl.py', 'rtl/board/board_ctrl.sv', {
        'tx_bit_period_off_by_one': ('rtl/board/uart_tx.sv', "sh <= {1'b1, sh[9:1]}; nbit <= nbit - 1'b1; cnt <= CW'(CPB - 1);", "sh <= {1'b1, sh[9:1]}; nbit <= nbit - 1'b1; cnt <= CW'(CPB);"),
        'rx_seven_data_bits': ('rtl/board/uart_rx.sv', "if (nbit == 4'd7) st <= S_STOP;", "if (nbit == 4'd6) st <= S_STOP;"),
        'rx_ignores_stop_bit': ('rtl/board/uart_rx.sv', "if (r1) begin data <= sh; valid <= 1'b1; end", "begin data <= sh; valid <= 1'b1; end"),
        'response_logit_bytes_swapped': ('rtl', "resp[7] <= probe[7:0]; resp[8] <= probe[15:8];", "resp[7] <= probe[15:8]; resp[8] <= probe[7:0];"),
        'response_cycle_bytes_swapped': ('rtl', "resp[3] <= (cyc + 1'b1) >> 0; resp[4] <= (cyc + 1'b1) >> 8;", "resp[3] <= (cyc + 1'b1) >> 8; resp[4] <= (cyc + 1'b1) >> 0;"),
        'request_token_bytes_swapped': ('rtl', "dec_token <= {fb[1], fb[0]};", "dec_token <= {fb[0], fb[1]};"),
        'request_pos_from_wrong_byte': ('rtl', "dec_pos <= fb[2][PW-1:0];", "dec_pos <= fb[3][PW-1:0];"),
        'probe_index_off_by_one': ('rtl', "dec_logit_idx == pidx)", "dec_logit_idx == pidx + 16'd1)"),
        'spi_response_one_byte_short': ('rtl', "rlen <= 4'(2 + nrx);", "rlen <= 4'(1 + nrx);"),
        'spi_clocks_one_bit_short': ('rtl', "bit_n <= {fb[0], 3'b000} + {fb[1], 3'b000};", "bit_n <= {fb[0], 3'b000} + {fb[1], 3'b000} - 9'd1;"),
        'frame_timeout_never_fires': ('rtl', "if (tmo == 25'(TMO)) st <= S_IDLE;", "if (1'b0) st <= S_IDLE;"),
        'response_before_done': ('rtl', "if (dec_done) begin", "if (dec_done || cyc == 32'd5000) begin"),
        'cs_not_released_after_spi': ('rtl', "3'd5: if (hc == 8'd3) begin cs_n_q <= 1'b1;", "3'd5: if (hc == 8'd3) begin cs_n_q <= 1'b0;",
                                      'equivalent at the pins: when the S transaction ends the pins go back to qspi_reader, whose CS# is high (idle: CS# rises there, 8 cycles late) or low (burst parked by backpressure: CS# stays low with SCLK low and no clocks). The CS# rise that latches WREN/WRDI/WRR is only deferred to the next CS# rise, which always comes before any further command byte is clocked: the next S frame forces CS# high first (S_PT_WAIT sets cs_n_q=1 before pt_active) and a decode seek raises it in qspi_reader. WEL/WIP are only observed through S frames or take effect before the next decode read, so the deferral is unobservable; the directed WREN/RDSR/WRDI/RDSR cases between decode steps (reader parked with CS# low) pass with the fault'),
        'flash_image_offset_wrong': ('rtl/board/boolean_core.sv', "parameter logic [31:0] FLASH_BASE = 32'h0028_0000,", "parameter logic [31:0] FLASH_BASE = 32'h0027_0000,"),
        'cycle_count_off_by_one': ('rtl', "resp[3] <= (cyc + 1'b1) >> 0;", "resp[3] <= (cyc + 2'd2) >> 0;"),
        # robustness additions (error frames, sequence rule, watchdog, full-width position path); detected by tools/board/tb/board_ctrl_unit_tb.sv
        # (run by the runner in --fast mode as well) or, for faults in boolean_core.sv, by the stalled-stream robustness run
        'token_bound_removed': ('rtl', "wire tok_bad = {fb[1], fb[0]} >= 16'(VOCAB);", "wire tok_bad = 1'b0;"),
        'pos_bound_removed': ('rtl', "wire pos_bad = pos_in >= 9'(MAX_SEQ);", "wire pos_bad = 1'b0;"),
        'sequence_gap_check_removed': ('rtl', "wire gap_bad = pos_in > next_pos;", "wire gap_bad = 1'b0;"),
        'watchdog_disabled': ('rtl', "if (!dec_done && cyc >= 32'(WDOG)) begin", "if (1'b0) begin"),
        'watchdog_abort_not_asserted': ('rtl', "dec_abort <= 1'b1; ab_cnt <= '0; st <= S_ABORT;", "ab_cnt <= '0; st <= S_ABORT;"),
        'watchdog_abort_too_short': ('rtl', "if (ab_cnt == 5'd16) begin dec_abort <= 1'b0;", "if (ab_cnt == 5'd2) begin dec_abort <= 1'b0;"),
        'pos_bit6_dropped': ('rtl', "dec_pos <= fb[2][PW-1:0];", "dec_pos <= {1'b0, fb[2][PW-2:0]};"),
        'next_pos_never_advances': ('rtl', "if ({1'b0, fb[2]} == next_pos) next_pos <= next_pos + 1'b1;", ""),
        'x_does_not_clear_next_pos': ('rtl', "8'h58: begin next_pos <= '0; resp[0]", "8'h58: begin resp[0]"),
        'reset_keeps_next_pos': ('rtl', "next_pos <= '0; dec_abort <= 1'b0; ab_cnt <= '0;", "dec_abort <= 1'b0; ab_cnt <= '0;"),
        'error_frame_wrong_code_for_pos': ('rtl', "else if (pos_bad) err(8'h02);", "else if (pos_bad) err(8'h03);"),
        'decode_not_reset_by_watchdog': ('rtl/board/boolean_core.sv', ".clk, .rst(rst_dec), .start(dec_start)", ".clk, .rst, .start(dec_start)"),
        'reader_not_reset_by_watchdog': ('rtl/board/boolean_core.sv', ".clk, .rst(rst_dec), .seek_req, .seek_addr(seek_addr + FLASH_BASE)", ".clk, .rst, .seek_req, .seek_addr(seek_addr + FLASH_BASE)",
                                         'equivalent under every stimulus the simulation can create: the stall is injected on the byte stream (st_valid forced low), which does not hang the reader, and a freshly reset top_decode issues seek_req for its first read, which makes the reader abort whatever burst it was in (newer address wins, unit qspi tests) and drop its pending byte. The reader reset in the watchdog path is defence in depth against a reader hang; a hang inside the reader cannot be produced without editing the reader, whose own unit tests and faults cover its seek/abort handling'),
        'spi_takeover_without_settle_wait': ('rtl', "if (pt_wait == 9'd255) begin", "if (pt_wait == 9'd0) begin",
                                             'equivalent: the 256-cycle wait only guards against taking the pins while the reader still clocks its last prefetched byte. The reader parks within about 16 cycles of the last byte top_decode accepts, which is before done; the host cannot send an S frame before it has received the 11-byte R response (11 x 10 bit times >= 880 cycles at 115200 baud, 880 cycles even at CPB=8 x 10 x 11) and the S header itself takes 3 more byte times, so the wait is always already elapsed in a protocol-conforming exchange. Removing it is unobservable'),
    }),
    'attn15': ('tools/decode/run_attn15_rtl.py', 'rtl/decode/attention.sv', {   # 15M geometry: bank aliasing wraps onto earlier positions only here
        'kv_bank_write_ge': ('rtl', "(32'(wfa) >> BAW) == bk) mem", "(32'(wfa) >> BAW) <= bk) mem"),
    }),
    'gemv': ('tools/decode/run_gemv_rtl.py', 'rtl/decode/gemv_unit.sv', {
        'ties_half_up': ('rtl', "inc  = (rem > half) || ((rem == half) && q[0]);", "inc  = (rem >= half);"),
        'no_div127_up_correction': ('rtl', "if (err2 > 66'sd127)       q5 <= q4 + 64'd1;", "if (err2 > 66'sd127)       q5 <= q4;",
                                    'equivalent: INV127_Q32 = 33818641 > 2^32/127, so the estimate never falls below the true quotient and the +1 branch is unreachable'),
        'no_div127_down_correction': ('rtl', "else if (err2 < -66'sd127) q5 <= q4 - 64'd1;", "else if (err2 < -66'sd127) q5 <= q4;"),
        'drop_last_beat_product': ('rtl', "snap_acc[i]   <= acc[i] + prod[i];", "snap_acc[i]   <= acc[i];"),
        'int4_nibble_swap': ('rtl', "w = nibble_hi ? {{4{b[7]}}, b[7:4]} : {{4{b[3]}}, b[3:0]};", "w = nibble_hi ? {{4{b[3]}}, b[3:0]} : {{4{b[7]}}, b[7:4]};"),
    }),
}


# PLACEHOLDER (DDR plumbing, host/runner/gate agent): the unit 'qspi_ddr' re-runs the QSPI unit test with --ddr (reader DDR=1, EDh, flash CR1 LC=11).
# The RTL agent adds the real EDh/DDR reader faults to this dict (same (file-kind, old, new[, equivalence reason]) tuples as 'qspi' above).
# While the dict is EMPTY, tools/decode_gate.py FAILS ("no faults defined") so the unit can never pass vacuously.
FAULTS['qspi_ddr'] = ('tools/decode/run_qspi_rtl.py --ddr', 'rtl/decode/qspi_reader.sv', {
    # 'ddr_fault_name': ('rtl', "old text", "new text"),
})


def run_fault(unit, runner, rtl, name, spec):
    """Build the mutant, run the unit's runner, classify. Returns the result dict for the fault."""
    which, old, new, *why = spec
    relsrc = PKG if which == 'pkg' else (rtl if which == 'rtl' else which)
    text = (ROOT / relsrc).read_text()
    assert text.count(old) == 1, (unit, name)
    mutant = ROOT / 'build/decode' / f'fault_{unit}_{name}.sv'
    mutant.write_text(text.replace(old, new))
    out = ROOT / 'build/decode' / f'fault_{unit}_{name}.json'
    out.unlink(missing_ok=True)                      # a stale artifact must never be mistaken for this run's
    script, *extra = runner.split()
    if unit == 'board':   # UART end-to-end runner in fast mode: the mutant replaces one source file
        args = [sys.executable, str(ROOT / script), '--fast', '--output', str(out), '--replace', f'{relsrc}={mutant}']
    elif unit == 'top':   # end-to-end runner: mutant top_decode.sv, mismatches = logits + indices + tokens
        args = [sys.executable, str(ROOT / script), *extra, '--output', str(out), '--top-rtl', str(mutant)]
    else:
        args = [sys.executable, str(ROOT / script), *extra, '--output', str(out), '--pkg' if which == 'pkg' else '--rtl', str(mutant)]
    res = {}
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        rc, tail = proc.returncode, (proc.stdout + proc.stderr)[-400:]
    except subprocess.TimeoutExpired:
        rc, tail = None, 'runner wall-clock timeout'
    o = None
    if out.exists():
        try:
            o = json.loads(out.read_text())
        except ValueError:
            o = None
    if unit == 'top' and o:
        lm, tm = o.get('logit_mismatches', -1), o.get('token_mismatches', -1)
        mm = (lm + tm + max(o.get('index_errors', 0), 0) + max(o.get('argmax_errors', 0), 0)) if min(lm, tm) >= 0 else -1
        real = mm
    elif o:
        mm, real = o.get('mismatch_count', -1), o.get('real_mismatch_count', -1)
    else:
        mm = real = -1
    # an error is: no artifact, a sentinel/negative count (the sim itself failed), or an exit code inconsistent with the status
    consistent = o is not None and rc is not None and ((rc == 0 and o.get('status') == 'pass') or (rc in (0, 1) and o.get('status') == 'fail'))   # rc 0 with fail: legacy runner not yet on finish()
    if not consistent or mm < 0 or mm >= SENTINEL or (real is not None and real >= SENTINEL):
        res = {'error': f'runner rc={rc}, artifact={"missing" if o is None else o.get("status")}, mismatches={mm}: {tail.strip()[-300:]}',
               'mismatches': mm, 'real_mismatches': real, 'detected': False, 'ok': False}
    else:
        res = {'mismatches': mm, 'real_mismatches': real, 'detected': mm > 0, 'runner_exit_code': rc}
        res['ok'] = res['detected'] != bool(why)
    if why:
        res['equivalent_reason'] = why[0]
    return res


def main():
    argv = sys.argv[1:]
    path = ROOT / 'bench/results/decode_fault_injection.json'
    if argv[:1] == ['--out']:
        path, argv = Path(argv[1]).resolve(), argv[2:]
    results, prov, all_ok, partial = {}, {}, True, set()
    for arg in argv:                                  # `unit` or `unit:fault1,fault2` (partial runs merge per fault)
        unit, _, only = arg.partition(':')
        if only:
            partial.add(unit)
        runner, rtl, faults = FAULTS[unit]
        res = {}
        for name, spec in faults.items():
            if only and name not in only.split(','):
                continue
            res[name] = run_fault(unit, runner, rtl, name, spec)
            all_ok &= res[name]['ok']
            print(unit, name, res[name], flush=True)
        results.setdefault(unit, {}).update(res)
        if not only:                               # a partial run does not refresh the unit's provenance
            prov[unit] = provenance(FAULT_SOURCES)
    merged = json.loads(path.read_text()) if path.exists() else {}   # re-read: other runs may have written
    for u, r in results.items():       # a full-unit run replaces the unit's entry; unit:fault runs update single faults
        merged[u] = {**merged.get(u, {}), **r} if u in partial else r
    merged.setdefault('provenance', {}).update(prov)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + '\n')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
