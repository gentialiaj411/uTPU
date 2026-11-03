// QSPI flash reader for top_decode's weight stream (target part: Spansion/Infineon S25FL128S, see
// docs/r3_progress.md "M4").
//
// Interface towards top_decode: seek_req (one-cycle pulse) + seek_addr in, st_valid/st_data out,
// st_ready in. After a seek, bytes are delivered in address order starting at seek_addr under a
// valid/ready handshake; st_valid is held low in the cycle seek_req is high.
//
// Flash side: Quad Output Read (6Bh with a 3-byte address, 6Ch with a 4-byte address), SPI mode 0.
// Instruction and address go out on IO0 (IO2/IO3 are driven high, IO1 is an input), then DUMMY
// clocks, then the flash drives IO[3:0] and a byte is two nibbles, high nibble first. Data changes
// on the flash's falling SCLK edge. SCLK = clk / (2*HALF).
//
// Backpressure: when the output byte register is occupied and the next rising edge would complete
// a second byte, SCLK is stopped low with CS# held (legal for SPI NOR); it resumes when the byte is
// taken. A seek raises CS# (waiting for SCLK low, and for a byte boundary inside the
// instruction/address/dummy phase so CS# is never raised mid-byte), idles CS# high CS_HIGH cycles,
// then issues a fresh command. Bytes assembled but not yet delivered are dropped on a seek.
//
// Zero BRAM, no vendor primitives: the pads/STARTUPE2 live in the board wrapper.
//
// DDR=1: DDR Quad I/O Read EDh instead (S25FL128S/S25FL256S datasheet 001-98283 Rev. *T, sec. 11.4.9 p. 104, Fig. 94
// "DDR Quad I/O Read initial access (3-byte address, EDh [ExtAdd = 0], HPLC = 11b)" p. 105; signal protocol sec. 4.3.20 to
// 4.3.24 p. 29-30). Requires CR1.QUAD = 1 and latency code CR1[7:6] = 11, which gives 1 mode cycle and 3 dummy cycles up to
// 50 MHz (Tables 27 and 29, p. 54-55, the same for EDh in the HPLC and EHPLC tables); DLP must be off (fewer than 5 dummy
// cycles, notes 45/46). Only HALF=1 (SCK 25 MHz at a 50 MHz core), ADDR_BYTES=3, SAMPLE_LATE=1 are accepted (guards below).
//   Instruction: EDh, 8 SCK cycles SDR on IO0 exactly as 6Bh (IO2/IO3 driven high, IO1 input, bits change on SCK falling).
//   Address + mode: 6 address nibbles A[23:20]..A[3:0], then the mode byte DDR_MODE high nibble first, on IO[3:0], one nibble
//     per SCK edge, the rising edge of each cycle first (Fig. 94: cycles 8-10 address, 11 mode; the nibbles sampled on the
//     rising edges are A[23:20], A[15:12], A[7:4], mode[7:4]). Mode 00h: the nibbles are not complementary (5h/Ah), so the
//     flash does not enter continuous mode and every command carries the EDh instruction (sec. 11.4.9).
//     Timing (FPGA side, not from the datasheet): in this phase SCK runs at half speed (half period 2 core cycles) and each
//     nibble changes on the core cycle BETWEEN two SCK edges, so it is stable one core cycle (20 ns) before and after the edge
//     that samples it. At full speed a nibble would have to change on the same core edge as an SCK edge, and the STARTUPE2
//     USRCCLKO -> CCLK delay (0.5..7.5 ns, constraints/boolean_xc7s50.xdc) would leave no guaranteed setup or hold (tSU/tHD
//     2 ns, Table 15). The flash only sees longer SCK phases (no minimum SCK frequency; tCH/tCL are minimums). A[23:20] is
//     the exception: it goes out with the falling edge that ends instruction cycle 7 (not a sampling edge for EDh), one core
//     cycle before the first address rising edge, like the first address bit of 6Bh. Cost: 8 core cycles per command.
//   Dummy: DDR_DUMMY = 3 SCK cycles at full speed; IO[3:0] are released at the first dummy rising edge, one core cycle after
//     the falling edge that samples the last mode nibble (so that nibble keeps its hold time): "High-Z Bus Turn-around".
//   Data: Fig. 94 cycle 15 (Data 0): the flash launches the high nibble [7:4] on the rising edge that starts cycle 15 (the
//     16th rising edge of the command) and the low nibble [3:0] on the falling edge after it, each valid tV after its edge
//     and held tHO after the next edge (Table 15, Fig. 41 "SPI DDR output timing"); the optional DLP would occupy the dummy
//     edges up to the falling edge that ends cycle 14, so that edge launches no data. So from rising edge 16 on every SCK
//     toggle launches one nibble (rising: high, falling: low); it is sampled one core cycle after the toggle (+ CAP_DLY),
//     i.e. on the core edge of the next toggle. (Checked against the independently written flash model.)
//   Backpressure: a rising toggle (it starts a byte; the falling toggle after it completes the byte and is never withheld) is
//     withheld unless that whole byte has a register, so SCLK stops low between bytes and an abort or reset never cuts a
//     byte in half. Reserving the register at the rising toggle needs a third byte register (sk1_*) to keep the full rate.
//     A byte completes every 2 core cycles and its capture lags the toggle by
//     1 + CAP_DLY cycles, so DDR adds a second byte register (sk_*) behind the output register; with one register the stall
//     rule would halve the rate.
//   Seek: CS# is raised only before the first SCK edge of a command or in the data phase, never during the address, mode
//     or dummy cycles ("CS# should not be driven HIGH during mode or dummy bits", sec. 11.4.9).
//
// STARTUP_PULSES: SCLK pulses with CS# high after reset. On 7-series the config flash clock only
// reaches the pin through STARTUPE2/USRCCLKO, and the first few USRCCLKO edges after configuration
// are not forwarded; clocking with CS# high is harmless and consumes them. An even number of
// toggles is used so SCLK idles low if all edges are forwarded.
module qspi_reader #(
    parameter int ADDR_BYTES     = 3,     // 3: opcode 6Bh, 4: opcode 6Ch
    parameter int DUMMY          = 8,     // dummy clocks after the address (S25FL128S, LC=00: 8)
    parameter int HALF           = 1,     // core clocks per SCLK half period
    parameter bit SAMPLE_LATE    = 1'b1,  // 1: sample IO just before the falling edge, 0: at the rising edge
    parameter int STARTUP_PULSES = 4,
    parameter int CS_HIGH        = 2,     // minimum CS# high cycles between commands (tCS read >= 10 ns)
    parameter int CAP_DLY        = 0,     // extra core cycles between the sampling tick and the data reaching io_i: 0 with
                                          // direct pads, 2 behind qspi_pad_regs (1 output + 1 input IOB register)
    parameter int DDR            = 0      // 0: Quad Output Read 6Bh/6Ch (SDR); 1: DDR Quad I/O Read EDh (flash CR1 LC=11, header)
) (
    input  logic        clk,
    input  logic        rst,
    // top_decode stream interface
    input  logic        seek_req,
    input  logic [31:0] seek_addr,
    input  logic        st_ready,
    output logic        st_valid,
    output logic [7:0]  st_data,
    // flash pins (pad primitives are outside)
    output logic        sclk,
    output logic        cs_n,
    output logic [3:0]  io_o,
    output logic [3:0]  io_oe,
    input  logic [3:0]  io_i
);
    // Elaboration guard: only parameter values exercised by tools/decode/run_qspi_rtl.py are accepted (audit N04: an untested
    // value, e.g. a bug that only appears at HALF=4, must not be silently buildable). Tested sets: ADDR_BYTES {3,4},
    // HALF {1,2,3,4}, DUMMY {0,8} (S25FL128S latency code 11 / 00), CAP_DLY {0,2}, STARTUP_PULSES {3,4} (UG470: the first
    // 3 USRCCLKO cycles are lost, so < 3 would corrupt the first command), CS_HIGH {2}, SAMPLE_LATE {0,1}. Extend the
    // test matrix before extending these sets. $error is ignored by some tools (iverilog 12), so an undefined module is
    // instantiated as well to force an elaboration failure everywhere.
    if (ADDR_BYTES != 3 && ADDR_BYTES != 4) begin : g_bad_addr_bytes
        $error("qspi_reader: ADDR_BYTES=%0d untested (3 or 4)", ADDR_BYTES);
        qspi_reader_untested_parameter_ADDR_BYTES u_bad ();
    end
    if (HALF < 1 || HALF > 4) begin : g_bad_half
        $error("qspi_reader: HALF=%0d untested (1..4)", HALF);
        qspi_reader_untested_parameter_HALF u_bad ();
    end
    if (DUMMY != 0 && DUMMY != 8) begin : g_bad_dummy
        $error("qspi_reader: DUMMY=%0d untested (0 or 8)", DUMMY);
        qspi_reader_untested_parameter_DUMMY u_bad ();
    end
    if (CAP_DLY != 0 && CAP_DLY != 2) begin : g_bad_cap_dly
        $error("qspi_reader: CAP_DLY=%0d untested (0 or 2)", CAP_DLY);
        qspi_reader_untested_parameter_CAP_DLY u_bad ();
    end
    if (STARTUP_PULSES != 3 && STARTUP_PULSES != 4) begin : g_bad_startup
        $error("qspi_reader: STARTUP_PULSES=%0d untested (3 or 4; UG470 needs >= 3 cycles)", STARTUP_PULSES);
        qspi_reader_untested_parameter_STARTUP_PULSES u_bad ();
    end
    if (CS_HIGH != 2) begin : g_bad_cs_high
        $error("qspi_reader: CS_HIGH=%0d untested (2)", CS_HIGH);
        qspi_reader_untested_parameter_CS_HIGH u_bad ();
    end
    // DDR {0,1}; DDR=1 only with HALF=1, ADDR_BYTES=3, SAMPLE_LATE=1, CAP_DLY {0,2} (above), and DUMMY left at its default 8
    // (DUMMY is the 6Bh dummy count and is not used by EDh, whose count is DDR_DUMMY).
    if (DDR != 0 && DDR != 1) begin : g_bad_ddr
        $error("qspi_reader: DDR=%0d untested (0 or 1)", DDR);
        qspi_reader_untested_parameter_DDR u_bad ();
    end
    if (DDR == 1 && HALF != 1) begin : g_bad_ddr_half
        $error("qspi_reader: DDR=1 with HALF=%0d untested (1)", HALF);
        qspi_reader_untested_parameter_DDR_HALF u_bad ();
    end
    if (DDR == 1 && ADDR_BYTES != 3) begin : g_bad_ddr_addr_bytes
        $error("qspi_reader: DDR=1 with ADDR_BYTES=%0d untested (3)", ADDR_BYTES);
        qspi_reader_untested_parameter_DDR_ADDR_BYTES u_bad ();
    end
    if (DDR == 1 && !SAMPLE_LATE) begin : g_bad_ddr_sample_late
        $error("qspi_reader: DDR=1 with SAMPLE_LATE=0 untested (1)");
        qspi_reader_untested_parameter_DDR_SAMPLE_LATE u_bad ();
    end
    if (DDR == 1 && DUMMY != 8) begin : g_bad_ddr_dummy
        $error("qspi_reader: DDR=1 with DUMMY=%0d (EDh uses DDR_DUMMY; leave DUMMY at 8)", DUMMY);
        qspi_reader_untested_parameter_DDR_DUMMY u_bad ();
    end

    localparam int SHW = 8 + 8 * ADDR_BYTES;
    localparam int PRE = SHW + DUMMY;                 // rising edges before the first data rising edge
    localparam logic [7:0] OPC = DDR ? 8'hED : (ADDR_BYTES == 4) ? 8'h6C : 8'h6B;
    localparam int TW = (HALF > 1) ? $clog2(HALF) : 1;
    // EDh (DDR=1), S25FL128S Rev. *T Tables 27/29, LC=11 at <= 50 MHz: 1 mode cycle, 3 dummy cycles
    localparam int DDR_DUMMY = 3;
    localparam logic [7:0] DDR_MODE = 8'h00;          // nibbles not complementary: no continuous read mode
    localparam int AMR = ADDR_BYTES + 1;              // SCK cycles of address + mode (one byte per cycle)
    localparam int DSW = 8 * AMR;                     // address + mode nibble shift register
    localparam int PRE_DDR = 8 + AMR + DDR_DUMMY;     // rising edges of instruction, address, mode and dummy (15); edge 16 launches Data 0
    localparam int PRX = DDR ? PRE_DDR + 1 : PRE;     // DDR: rise_n reaches 16 once Data 0's high nibble has been launched

    typedef enum logic [1:0] {S_START, S_IDLE, S_GAP, S_RUN} state_t;
    state_t        state;
    logic          sclk_q, cs_n_q;
    logic [3:0]    io_o_q, io_oe_q;
    logic [TW-1:0] tcnt;
    logic [7:0]    rise_n;           // rising edges issued in this command, saturates at PRE
    logic [SHW-1:0] sh;
    logic          pend;
    logic [31:0]   pend_addr;
    logic [7:0]    su_cnt;
    logic [3:0]    gcnt;
    logic          nib_ph, last_rise_data;
    logic [3:0]    nib_hi;
    logic [7:0]    ob_data;
    logic          ob_valid;
    localparam int CD = (CAP_DLY > 0) ? CAP_DLY : 1;
    logic [CD-1:0] cap_p;            // sampling strobes in flight (CAP_DLY > 0)
    // DDR only (constant in a DDR=0 build)
    logic [DSW-1:0] dsh;             // address + mode nibbles still to send
    logic          dph;              // address/mode phase: the nibble for the next SCK edge is already on IO
    logic          dl_q;             // the last core edge toggled SCK in the data phase (= the flash launched a nibble)
    logic [7:0]    sk_data;          // second byte register behind ob_data
    logic          sk_valid;
    logic [7:0]    sk1_data;         // third byte register behind sk_data
    logic          sk1_valid;

    assign sclk     = sclk_q;
    assign cs_n     = cs_n_q;
    assign io_o     = io_o_q;
    assign io_oe    = io_oe_q;
    assign st_valid = ob_valid && !seek_req;
    assign st_data  = ob_data;

    wire accept_now  = st_valid && st_ready;
    wire room        = !ob_valid || accept_now;
    wire data_phase  = (rise_n >= 8'(PRX));
    // DDR: this toggle launches a data nibble: rising edge 16 onwards (sclk low, rise_n = 15 before it), then every edge
    wire ddr_launch_ok = sclk_q ? (rise_n >= 8'(PRE_DDR + 1)) : (rise_n >= 8'(PRE_DDR));
    wire tick_raw    = (tcnt == TW'(HALF - 1));
    // nibble phase the next rising edge will sample, counting captures still in flight, and whether an in-flight capture
    // will complete a byte (it will occupy the output register before the next one can)
    logic eff_ph, pend_complete; logic [1:0] pend_n;
    always_comb begin
        eff_ph = nib_ph; pend_complete = 1'b0; pend_n = 2'd0;
        if (CAP_DLY > 0) for (int i = CAP_DLY - 1; i >= 0; i--) if (cap_p[i]) begin
            if (eff_ph) begin pend_complete = 1'b1; pend_n = pend_n + 2'd1; end
            eff_ph = ~eff_ph;
        end
        if (DDR && dl_q) begin                        // DDR: the nibble launched by the last toggle is in flight as well
            if (eff_ph) begin pend_complete = 1'b1; pend_n = pend_n + 2'd1; end
            eff_ph = ~eff_ph;
        end
    end
    // do not raise SCLK if the nibble it samples would complete a byte with nowhere to put it
    wire rise_blocked = !sclk_q && data_phase && eff_ph && !(room && !pend_complete);
    // DDR: same rule with two byte registers: the byte this rising toggle completes needs a register after every byte still
    // held after this cycle's handoff and every byte completion already in flight
    wire [1:0] occ_after = 2'(ob_valid) + 2'(sk_valid) + 2'(sk1_valid) - 2'(accept_now);
    // DDR: a rising toggle starts a byte; it needs a register for that byte after every byte still held after this cycle's
    // handoff and every completion already in flight (the falling toggle that completes it follows unconditionally)
    wire rise_blocked_ddr = ddr_launch_ok && !sclk_q && (3'(occ_after) + 3'(pend_n) > 3'd2);
    wire stop        = DDR ? rise_blocked_ddr : rise_blocked;
    wire tick        = tick_raw && !stop;
    wire abort_sdr   = (data_phase || rise_n[2:0] == 3'd0);
    wire abort_ddr   = (data_phase || rise_n == 8'd0);  // EDh: never inside the address, mode or dummy cycles
    wire abort_now   = pend && !sclk_q && (DDR ? abort_ddr : abort_sdr);
    // DDR address/mode phase: SCK edges after rise 8 up to the falling edge after rise 8+AMR. A tick first changes the nibble
    // (dph=0), the next one toggles SCK (dph=1).
    wire ddr_am      = DDR && (rise_n >= 8'd9) && (sclk_q ? (rise_n <= 8'(8 + AMR)) : (rise_n < 8'(8 + AMR)));
    wire dchg        = ddr_am && !dph;
    wire launch_sdr  = (state == S_RUN) && !abort_now && tick &&
                       (SAMPLE_LATE ? (sclk_q && last_rise_data) : (!sclk_q && data_phase));
    // DDR: the nibble launched by the toggle at core edge E is sampled at edge E+1 (direct pads; + CAP_DLY behind pad registers)
    wire launch      = DDR ? dl_q : launch_sdr;
    wire strobe      = (CAP_DLY == 0) ? launch : cap_p[CD-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_START; sclk_q <= 1'b0; cs_n_q <= 1'b1; io_o_q <= 4'hF; io_oe_q <= 4'h0;
            cap_p <= '0; tcnt <= '0; rise_n <= '0; pend <= 1'b0; su_cnt <= 8'(2 * STARTUP_PULSES); gcnt <= '0;
            nib_ph <= 1'b0; last_rise_data <= 1'b0; ob_valid <= 1'b0;
            sk_valid <= 1'b0; sk1_valid <= 1'b0; dl_q <= 1'b0; dph <= 1'b0;
        end else begin
            if (accept_now) ob_valid <= 1'b0;
            if (accept_now && sk_valid) begin                               // DDR only: the queue moves up one register
                ob_data <= sk_data; ob_valid <= 1'b1;
                if (sk1_valid) begin sk_data <= sk1_data; sk1_valid <= 1'b0; end else sk_valid <= 1'b0;
            end
            cap_p <= (cap_p << 1) | CD'(launch);
            dl_q <= DDR && state == S_RUN && !abort_now && tick && !dchg && ddr_launch_ok;
            case (state)
                S_START: begin
                    if (su_cnt == 8'd0) state <= S_IDLE;
                    else if (tick_raw) begin sclk_q <= ~sclk_q; su_cnt <= su_cnt - 1'b1; tcnt <= '0; end
                    else tcnt <= tcnt + 1'b1;
                end
                S_IDLE: if (pend) begin state <= S_GAP; gcnt <= 4'(CS_HIGH - 1); end
                S_GAP: begin
                    if (seek_req) ;                               // newer address wins, restart nothing
                    else if (gcnt != 4'd0) gcnt <= gcnt - 1'b1;
                    else begin
                        state <= S_RUN; cs_n_q <= 1'b0; pend <= 1'b0;
                        sh <= {OPC, pend_addr[8*ADDR_BYTES-1:0]};
                        io_o_q <= {3'b111, OPC[7]}; io_oe_q <= 4'b1101;
                        rise_n <= '0; tcnt <= '0; nib_ph <= 1'b0; last_rise_data <= 1'b0; ob_valid <= 1'b0; cap_p <= '0;
                        dsh <= {pend_addr[8*ADDR_BYTES-1:0], DDR_MODE}; dph <= 1'b0; sk_valid <= 1'b0; sk1_valid <= 1'b0;
                    end
                end
                S_RUN: begin
                    if (abort_now) begin
                        state <= S_GAP; gcnt <= 4'(CS_HIGH - 1); cs_n_q <= 1'b1; io_oe_q <= 4'h0; io_o_q <= 4'hF;
                        ob_valid <= 1'b0; nib_ph <= 1'b0; tcnt <= '0; cap_p <= '0;
                        sk_valid <= 1'b0; sk1_valid <= 1'b0;
                    end else begin
                        if (strobe) cap(io_i);
                        if (tick_raw) begin if (!stop) tcnt <= '0; end
                        else tcnt <= tcnt + 1'b1;
                        if (tick && dchg) begin                       // DDR: next address/mode nibble, SCK holds
                            io_o_q <= dsh[DSW-1 -: 4]; dsh <= dsh << 4; dph <= 1'b1;
                        end else if (tick) begin
                            sclk_q <= ~sclk_q; dph <= 1'b0;
                            if (!sclk_q) begin                        // rising edge
                                if (rise_n != 8'(PRX)) rise_n <= rise_n + 1'b1;
                                last_rise_data <= data_phase;
                                if (DDR && rise_n == 8'(8 + AMR)) io_oe_q <= 4'h0;   // DDR first dummy rise: turn-around
                            end else if (DDR) begin                   // DDR falling edge: instruction bit, or A[23:20]
                                if (rise_n < 8'd8) begin
                                    sh <= sh << 1; io_o_q[0] <= sh[SHW-2];
                                end else if (rise_n == 8'd8) begin
                                    io_o_q <= dsh[DSW-1 -: 4]; dsh <= dsh << 4; io_oe_q <= 4'hF;
                                end
                            end else begin                            // falling edge: launch next bit
                                if (rise_n < 8'(SHW)) begin
                                    sh <= sh << 1; io_o_q[0] <= sh[SHW-2];
                                end else begin
                                    io_oe_q <= 4'h0;
                                end
                            end
                        end
                    end
                end
            endcase
            if (seek_req) begin pend <= 1'b1; pend_addr <= seek_addr; ob_valid <= 1'b0; end
            if (seek_req) begin sk_valid <= 1'b0; sk1_valid <= 1'b0; end
        end
    end

    // assemble nibbles into the output byte register, high nibble first. DDR: a completed byte goes to the first register that
    // is free after this cycle's handoff and queue shift (output register, then the second, then the third).
    task automatic cap(input logic [3:0] nib);
        if (!seek_req && !pend) begin
            if (!nib_ph) begin nib_hi <= nib; nib_ph <= 1'b1; end
            else if (DDR && occ_after >= 2'd2) begin
`ifdef QSPI_MUTATE
                sk1_data <= {nib, nib_hi};
`else
                sk1_data <= {nib_hi, nib};
`endif
                sk1_valid <= 1'b1; nib_ph <= 1'b0;
            end else if (DDR && occ_after == 2'd1) begin
`ifdef QSPI_MUTATE
                sk_data <= {nib, nib_hi};
`else
                sk_data <= {nib_hi, nib};
`endif
                sk_valid <= 1'b1; nib_ph <= 1'b0;
            end else begin
`ifdef QSPI_MUTATE
                ob_data <= {nib, nib_hi};
`else
                ob_data <= {nib_hi, nib};
`endif
                ob_valid <= 1'b1; nib_ph <= 1'b0;
            end
        end
    endtask
endmodule
