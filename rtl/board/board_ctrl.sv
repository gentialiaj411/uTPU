// UART command front end for the Boolean top. Multi-byte fields are little endian.
//
//   'T' (54h) tok_lo tok_hi pos idx_lo idx_hi    run one decode step (token at position pos), probe logit idx
//     -> 'R' (52h) tok_lo tok_hi cyc[0..3] logit[0..3]
//        tok = argmax token, cyc = core cycles from start to done, logit = INT32 logit at idx (0 if idx >= VOCAB)
//   'S' (53h) ntx nrx tx[0..ntx-1]                single-bit SPI to the flash (ntx, nrx <= 8): CS# low, ntx bytes out
//                                                 on IO0, nrx bytes in on IO1, CS# high (flash register access)
//     -> 's' (73h) nrx rx[0..nrx-1]
//   'V' (56h)                                     -> 'v' (76h) 01h    (link check)
// Other bytes are ignored; a partial frame is dropped after TMO idle cycles. Bytes that arrive while a step or an
// SPI transaction runs are ignored, so the host must wait for the response before sending the next frame.
module board_ctrl #(parameter int PW = 7, PT_HALF = 2, TMO = 16777216) (
    input  logic          clk, rst,
    input  logic [7:0]    rx_data,
    input  logic          rx_valid,
    output logic [7:0]    tx_data,
    output logic          tx_start,
    input  logic          tx_busy,
    output logic          dec_start,
    output logic [15:0]   dec_token,
    output logic [PW-1:0] dec_pos,
    input  logic          dec_done,
    input  logic [15:0]   dec_token_out,
    input  logic          dec_logit_valid,
    input  logic [15:0]   dec_logit_idx,
    input  logic signed [31:0] dec_logit_val,
    output logic          pt_active, pt_sclk, pt_cs_n, pt_mosi,
    input  logic          pt_miso
);
    typedef enum logic [3:0] {S_IDLE, S_GATHER, S_START, S_RUN, S_PT_WAIT, S_PT, S_TX, S_TXW} st_t;
    st_t st;
    logic [7:0]  cmd;
    logic [7:0]  fb [10];
    logic [3:0]  have;
    logic [24:0] tmo;
    logic [31:0] cyc, probe;
    logic [15:0] pidx;
    logic [7:0]  resp [12];
    logic [3:0]  rlen, rptr;
    // SPI engine
    logic [7:0]  ntx, nrx;
    logic [7:0]  rb [8];
    logic [8:0]  bit_i, bit_n, pt_wait;
    logic [2:0]  pstate;
    logic [7:0]  hc;
    logic        cs_n_q, sclk_q;
    assign pt_cs_n = cs_n_q;
    assign pt_sclk = sclk_q;
    wire [8:0] tx_bits = {ntx, 3'b000} >> 0;
    assign pt_mosi = (bit_i < tx_bits) ? fb[2 + 4'(bit_i >> 3)][3'd7 - bit_i[2:0]] : 1'b0;

    wire [3:0] nh        = have + 1'b1;
    wire       is_s      = (cmd == 8'h53);
    wire [3:0] need_now  = is_s ? ((nh >= 4'd2) ? 4'd2 + fb[0][3:0] : 4'd2) : 4'd5;
    wire       s_bad     = is_s && nh == 4'd2 && (fb[0] > 8'd8 || rx_data > 8'd8);
    wire [8:0] rx_pos    = bit_i - tx_bits;

    always_ff @(posedge clk) begin
        dec_start <= 1'b0; tx_start <= 1'b0;
        if (rst) begin
            st <= S_IDLE; have <= '0; pt_active <= 1'b0; cs_n_q <= 1'b1; sclk_q <= 1'b0; pt_wait <= '0;
        end else begin
            if (dec_logit_valid && dec_logit_idx == pidx) probe <= dec_logit_val;
            case (st)
                S_IDLE: begin
                    pt_active <= 1'b0; cs_n_q <= 1'b1; sclk_q <= 1'b0; have <= '0; tmo <= '0; pt_wait <= '0;
                    if (rx_valid) begin
                        cmd <= rx_data;
                        case (rx_data)
                            8'h54, 8'h53: st <= S_GATHER;
                            8'h56: begin resp[0] <= 8'h76; resp[1] <= 8'h01; rlen <= 4'd2; rptr <= '0; st <= S_TX; end
                            default: ;
                        endcase
                    end
                end
                S_GATHER: begin
                    tmo <= tmo + 1'b1;
                    if (tmo == 25'(TMO)) st <= S_IDLE;
                    else if (rx_valid) begin
                        fb[have] <= rx_data; have <= nh; tmo <= '0;
                        if (s_bad) st <= S_IDLE;
                        else if (nh == need_now && (!is_s || nh >= 4'd2)) st <= is_s ? S_PT_WAIT : S_START;
                    end
                end
                S_START: begin
                    dec_token <= {fb[1], fb[0]}; dec_pos <= fb[2][PW-1:0]; pidx <= {fb[4], fb[3]};
                    probe <= '0; cyc <= '0; dec_start <= 1'b1; st <= S_RUN;
                end
                S_RUN: begin
                    cyc <= cyc + 1'b1;
                    if (dec_done) begin
                        resp[0] <= 8'h52;
`ifdef BOARD_MUTATE
                        resp[1] <= dec_token_out[15:8]; resp[2] <= dec_token_out[7:0];
`else
                        resp[1] <= dec_token_out[7:0]; resp[2] <= dec_token_out[15:8];
`endif
                        resp[3] <= (cyc + 1'b1) >> 0; resp[4] <= (cyc + 1'b1) >> 8; resp[5] <= (cyc + 1'b1) >> 16; resp[6] <= (cyc + 1'b1) >> 24;
                        resp[7] <= probe[7:0]; resp[8] <= probe[15:8]; resp[9] <= probe[23:16]; resp[10] <= probe[31:24];
                        rlen <= 4'd11; rptr <= '0; st <= S_TX;
                    end
                end
                // SPI passthrough: let the decode reader's tail clocks finish before the pins are taken over
                S_PT_WAIT: begin
                    ntx <= fb[0]; nrx <= fb[1]; pt_wait <= pt_wait + 1'b1;
                    if (pt_wait == 9'd255) begin
                        pt_active <= 1'b1; cs_n_q <= 1'b1; sclk_q <= 1'b0; bit_i <= '0; hc <= '0; pstate <= 3'd0;
                        bit_n <= {fb[0], 3'b000} + {fb[1], 3'b000}; st <= S_PT;
                        for (int k = 0; k < 8; k++) rb[k] <= 8'h00;
                    end
                end
                S_PT: begin
                    hc <= hc + 1'b1;
                    case (pstate)
                        3'd0: if (hc == 8'd3) begin cs_n_q <= 1'b0; hc <= '0; pstate <= 3'd1; end    // CS# low, wait tCSS
                        3'd1: if (hc == 8'(PT_HALF - 1)) begin                                      // SCLK low half
                                  hc <= '0;
                                  if (bit_i == bit_n) pstate <= 3'd5; else begin sclk_q <= 1'b1; pstate <= 3'd2; end
                              end
                        3'd2: if (hc == 8'(PT_HALF - 1)) begin                                      // SCLK high half; sample before the fall
                                  if (bit_i >= tx_bits) rb[3'(rx_pos >> 3)] <= {rb[3'(rx_pos >> 3)][6:0], pt_miso};
                                  sclk_q <= 1'b0; bit_i <= bit_i + 1'b1; hc <= '0; pstate <= 3'd1;
                              end
                        3'd5: if (hc == 8'd3) begin cs_n_q <= 1'b1; hc <= '0; pstate <= 3'd6; end   // CS# high
                        3'd6: if (hc == 8'd7) begin
                                  resp[0] <= 8'h73; resp[1] <= nrx;
                                  for (int k = 0; k < 8; k++) resp[2 + k] <= rb[k];
                                  rlen <= 4'(2 + nrx); rptr <= '0; pt_active <= 1'b0; st <= S_TX;
                              end
                        default: ;
                    endcase
                end
                S_TX: if (!tx_busy) begin tx_data <= resp[rptr]; tx_start <= 1'b1; st <= S_TXW; end
                S_TXW: begin
                    if (rptr + 1'b1 == rlen) st <= S_IDLE;
                    else begin rptr <= rptr + 1'b1; st <= S_TX; end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
