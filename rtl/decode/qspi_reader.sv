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
    parameter int CAP_DLY        = 0      // extra core cycles between the sampling tick and the data reaching io_i: 0 with
                                          // direct pads, 2 behind qspi_pad_regs (1 output + 1 input IOB register)
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
    localparam int SHW = 8 + 8 * ADDR_BYTES;
    localparam int PRE = SHW + DUMMY;                 // rising edges before the first data rising edge
    localparam logic [7:0] OPC = (ADDR_BYTES == 4) ? 8'h6C : 8'h6B;
    localparam int TW = (HALF > 1) ? $clog2(HALF) : 1;

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

    assign sclk     = sclk_q;
    assign cs_n     = cs_n_q;
    assign io_o     = io_o_q;
    assign io_oe    = io_oe_q;
    assign st_valid = ob_valid && !seek_req;
    assign st_data  = ob_data;

    wire accept_now  = st_valid && st_ready;
    wire room        = !ob_valid || accept_now;
    wire data_phase  = (rise_n >= 8'(PRE));
    wire tick_raw    = (tcnt == TW'(HALF - 1));
    // nibble phase the next rising edge will sample, counting captures still in flight, and whether an in-flight capture
    // will complete a byte (it will occupy the output register before the next one can)
    logic eff_ph, pend_complete;
    always_comb begin
        eff_ph = nib_ph; pend_complete = 1'b0;
        if (CAP_DLY > 0) for (int i = CAP_DLY - 1; i >= 0; i--) if (cap_p[i]) begin
            if (eff_ph) pend_complete = 1'b1;
            eff_ph = ~eff_ph;
        end
    end
    // do not raise SCLK if the nibble it samples would complete a byte with nowhere to put it
    wire rise_blocked = !sclk_q && data_phase && eff_ph && !(room && !pend_complete);
    wire tick        = tick_raw && !rise_blocked;
    wire abort_now   = pend && !sclk_q && (data_phase || rise_n[2:0] == 3'd0);
    wire launch      = (state == S_RUN) && !abort_now && tick &&
                       (SAMPLE_LATE ? (sclk_q && last_rise_data) : (!sclk_q && data_phase));
    wire strobe      = (CAP_DLY == 0) ? launch : cap_p[CD-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_START; sclk_q <= 1'b0; cs_n_q <= 1'b1; io_o_q <= 4'hF; io_oe_q <= 4'h0;
            cap_p <= '0; tcnt <= '0; rise_n <= '0; pend <= 1'b0; su_cnt <= 8'(2 * STARTUP_PULSES); gcnt <= '0;
            nib_ph <= 1'b0; last_rise_data <= 1'b0; ob_valid <= 1'b0;
        end else begin
            if (accept_now) ob_valid <= 1'b0;
            cap_p <= (cap_p << 1) | CD'(launch);
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
                    end
                end
                S_RUN: begin
                    if (abort_now) begin
                        state <= S_GAP; gcnt <= 4'(CS_HIGH - 1); cs_n_q <= 1'b1; io_oe_q <= 4'h0; io_o_q <= 4'hF;
                        ob_valid <= 1'b0; nib_ph <= 1'b0; tcnt <= '0; cap_p <= '0;
                    end else begin
                        if (strobe) cap(io_i);
                        if (tick_raw) begin if (!rise_blocked) tcnt <= '0; end
                        else tcnt <= tcnt + 1'b1;
                        if (tick) begin
                            sclk_q <= ~sclk_q;
                            if (!sclk_q) begin                        // rising edge
                                if (rise_n != 8'(PRE)) rise_n <= rise_n + 1'b1;
                                last_rise_data <= data_phase;
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
        end
    end

    // assemble nibbles into the output byte register, high nibble first
    task automatic cap(input logic [3:0] nib);
        if (!seek_req && !pend) begin
            if (!nib_ph) begin nib_hi <= nib; nib_ph <= 1'b1; end
            else begin
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
