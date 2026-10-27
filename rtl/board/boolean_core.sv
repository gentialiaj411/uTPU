// Board core: UART command front end + top_decode + qspi_reader, no vendor primitives (simulates in iverilog).
// boolean_top.sv adds the MMCM, STARTUPE2/pads and the reset. Defaults are the stories15M configuration
// (identical to build/decode/decode_image_15m/params.json, checked by tools/board/impl_top.py); the UART test
// overrides them for stories260K.
//
// Flash map: the weight image starts at FLASH_BASE (default 0x280000 = 2.5 MiB, above the xc7s50 bitstream of
// 17,536,096 bits = 2,192,012 B); top_decode's image addresses are offset by it. The image budget
// 16 MiB - 2.5 MiB = 14,155,776 B.
module boolean_core #(
    parameter int CPB = 434,                       // UART clocks per bit (50 MHz / 115200)
    parameter int PT_HALF = 2,                     // flash register-access SCLK half period in core clocks
    parameter int TMO = 16777216,                  // UART partial-frame timeout, core clocks
    parameter logic [31:0] FLASH_BASE = 32'h0028_0000,
    parameter int QSPI_HALF = 1,
    parameter bit QSPI_SAMPLE_LATE = 1'b1,
    parameter int QSPI_CAP_DLY = 2,                // qspi_pad_regs adds 1 output + 1 input register
    parameter int D = 288, H = 768, NL = 6, NH = 6, NKV = 6, HS = 48, VOCAB = 32000, MAX_SEQ = 128,
    parameter int KV4 = 1,
    parameter int CLS4 = 1, WQ4 = 0, WK4 = 0, WV4 = 0, WO4 = 0, W14 = 1, W34 = 1, W24 = 0,
    parameter longint CLS_BASE = 0, EMB_SCALE_BASE = 6048000, LAYERS_BASE = 7488000, IMAGE_BYTES = 13073472,
    parameter int CLS_TILE_BYTES = 3024, CLS_KB = 144, CLS_G = 9,
    parameter longint RECIP_D = 14913081, NORM_MULT = 828896808, EPS = 1034,
    parameter int GFB = 12, NORM_SHIFT = 34,
    parameter longint SILU_MULT = 884304146, SWIGLU_MULT = 742221392, SCORE_MULT = 801779553,
    parameter int SILU_SHIFT = 37, SWIGLU_SHIFT = 43, SCORE_SHIFT = 50,
    parameter ROPE_CS = "rtl/decode/luts/rope_cs_q15_h48_p128.mem",
    parameter int ROPE_DEPTH = 3072
) (
    input  logic       clk, rst,
    input  logic       uart_rx,
    output logic       uart_tx,
    // pad side of the flash interface (registered in qspi_pad_regs; pad_io_t = 1 releases the pad)
    output logic       pad_sclk, pad_cs_n,
    output logic [3:0] pad_io_o, pad_io_t,
    input  logic [3:0] pad_io_i
);
    localparam int PW = $clog2(MAX_SEQ);
    // UART
    logic [7:0] rx_data, tx_data; logic rx_valid, tx_start, tx_busy;
    uart_rx #(.CPB(CPB)) u_rx (.clk, .rst, .rxd(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CPB(CPB)) u_tx (.clk, .rst, .start(tx_start), .data(tx_data), .txd(uart_tx), .busy(tx_busy));
    // decode engine + flash reader
    logic dec_start, dec_busy, dec_done, dec_logit_valid, seek_req, st_valid, st_ready;
    logic [15:0] dec_token, dec_token_out, dec_logit_idx; logic [PW-1:0] dec_pos;
    logic signed [31:0] dec_logit_val; logic [31:0] seek_addr; logic [7:0] st_data;
    top_decode #(.D(D), .H(H), .NL(NL), .NH(NH), .NKV(NKV), .HS(HS), .VOCAB(VOCAB), .MAX_SEQ(MAX_SEQ), .KV4(KV4),
        .CLS4(CLS4), .WQ4(WQ4), .WK4(WK4), .WV4(WV4), .WO4(WO4), .W14(W14), .W34(W34), .W24(W24),
        .CLS_BASE(CLS_BASE), .EMB_SCALE_BASE(EMB_SCALE_BASE), .LAYERS_BASE(LAYERS_BASE), .IMAGE_BYTES(IMAGE_BYTES),
        .CLS_TILE_BYTES(CLS_TILE_BYTES), .CLS_KB(CLS_KB), .CLS_G(CLS_G), .RECIP_D(RECIP_D), .NORM_MULT(NORM_MULT), .EPS(EPS),
        .GFB(GFB), .NORM_SHIFT(NORM_SHIFT), .SILU_MULT(SILU_MULT), .SWIGLU_MULT(SWIGLU_MULT), .SCORE_MULT(SCORE_MULT),
        .SILU_SHIFT(SILU_SHIFT), .SWIGLU_SHIFT(SWIGLU_SHIFT), .SCORE_SHIFT(SCORE_SHIFT), .ROPE_CS(ROPE_CS), .ROPE_DEPTH(ROPE_DEPTH)) u_dec (
        .clk, .rst, .start(dec_start), .token(dec_token), .pos(dec_pos), .seek_req, .seek_addr, .st_valid, .st_data, .st_ready,
        .busy(dec_busy), .done(dec_done), .token_out(dec_token_out), .logit_valid(dec_logit_valid), .logit_idx(dec_logit_idx),
        .logit_val(dec_logit_val));
    logic r_sclk, r_cs_n; logic [3:0] r_io_o, r_io_oe, io_i;
    qspi_reader #(.HALF(QSPI_HALF), .SAMPLE_LATE(QSPI_SAMPLE_LATE), .CAP_DLY(QSPI_CAP_DLY)) u_rd (
        .clk, .rst, .seek_req, .seek_addr(seek_addr + FLASH_BASE), .st_ready, .st_valid, .st_data,
        .sclk(r_sclk), .cs_n(r_cs_n), .io_o(r_io_o), .io_oe(r_io_oe), .io_i);
    // UART control + flash register access (single-bit SPI) multiplexed onto the flash pins
    logic pt_active, pt_sclk, pt_cs_n, pt_mosi;
    board_ctrl #(.PW(PW), .PT_HALF(PT_HALF), .TMO(TMO)) u_ctl (
        .clk, .rst, .rx_data, .rx_valid, .tx_data, .tx_start, .tx_busy,
        .dec_start, .dec_token, .dec_pos, .dec_done, .dec_token_out, .dec_logit_valid, .dec_logit_idx, .dec_logit_val,
        .pt_active, .pt_sclk, .pt_cs_n, .pt_mosi, .pt_miso(io_i[1]));
    logic sclk, cs_n; logic [3:0] io_o, io_oe;
    assign sclk  = pt_active ? pt_sclk : r_sclk;
    assign cs_n  = pt_active ? pt_cs_n : r_cs_n;
    assign io_o  = pt_active ? {2'b11, 1'b0, pt_mosi} : r_io_o;   // IO2/IO3 (WP#/HOLD#) high in single-bit mode
    assign io_oe = pt_active ? 4'b1101 : r_io_oe;
    qspi_pad_regs u_pr (.clk, .sclk, .cs_n, .io_o, .io_oe, .pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i, .io_i);
endmodule
