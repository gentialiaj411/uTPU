// Boolean board wrapper. The oscillator remains at 100 MHz; only core_clk is
// multiplied/divided by the MMCM. Simulation replay continues to instantiate
// `top` directly, keeping the MMCM primitive out of the Icarus source set.
module top_board #(
    parameter integer CORE_CLK_HZ = 50000000,
    parameter integer RST_ACTIVE_LOW = 1,
    parameter integer UART_BAUD = 115200,
    parameter integer FORCE_UART_ECHO = 0,
    parameter integer COMPUTE_DATA_WIDTH = 8,
    parameter integer ACCUMULATOR_DATA_WIDTH = 32,
    parameter integer ARRAY_SIZE = 8,
    parameter integer EXT_ADDR_EN = 1,
    parameter integer VECTOR_UNIT_EN = 0,
    parameter integer QUANTIZER_LANES = ARRAY_SIZE,
    parameter integer RELU_LANES = ARRAY_SIZE,
    parameter integer QUANTIZER_PIPE_DEPTH = 3,
    parameter integer WEIGHT_OVERLAP_EN = 0,
    parameter integer MAX_BATCH_COUNT = 4,
    parameter integer BUFFER_SIZE = 32768,
    parameter integer WEIGHT_REGION_WORDS = (BUFFER_SIZE < 14144) ? BUFFER_SIZE : 14144,
    parameter integer PROG_DEPTH = 32768
) (
    input  logic clk,
    input  logic rst,
    input  logic rx,
    output logic tx,
    output logic led_rst
);
    localparam real MMCM_VCO_HZ = 800000000.0;
    localparam real MMCM_OUT_DIVIDE = MMCM_VCO_HZ / CORE_CLK_HZ;

    wire mmcm_clk;
    wire mmcm_fb;
    wire mmcm_fb_buf;
    wire core_clk;
    wire mmcm_locked;
    wire reset_button_active = RST_ACTIVE_LOW ? ~rst : rst;
    wire reset_async_active = reset_button_active | ~mmcm_locked;
    logic reset_sync_meta = 1'b1;
    logic reset_sync_active = 1'b1;
    wire core_reset = RST_ACTIVE_LOW ? ~reset_sync_active : reset_sync_active;

    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKIN1_PERIOD(10.0),
        .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(8.0),
        .CLKOUT0_DIVIDE_F(MMCM_OUT_DIVIDE),
        .STARTUP_WAIT("FALSE")
    ) u_mmcm (
        .CLKIN1(clk),
        .CLKFBIN(mmcm_fb_buf),
        .CLKFBOUT(mmcm_fb),
        .CLKOUT0(mmcm_clk),
        .LOCKED(mmcm_locked),
        .PWRDN(1'b0),
        .RST(1'b0)
    );

    BUFG u_mmcm_fb_bufg (.I(mmcm_fb), .O(mmcm_fb_buf));
    BUFG u_core_clk_bufg (.I(mmcm_clk), .O(core_clk));

    // Asynchronous assertion, synchronous deassertion. The core remains held
    // until both the physical reset is released and the MMCM reports lock.
    always_ff @(posedge core_clk or posedge reset_async_active) begin
        if (reset_async_active) begin
            reset_sync_meta <= 1'b1;
            reset_sync_active <= 1'b1;
        end else begin
            reset_sync_meta <= 1'b0;
            reset_sync_active <= reset_sync_meta;
        end
    end

    top #(
        .UART_INPUT_CLK(CORE_CLK_HZ),
        .UART_BAUD(UART_BAUD),
        .RST_ACTIVE_LOW(RST_ACTIVE_LOW),
        .FORCE_UART_ECHO(FORCE_UART_ECHO),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .ARRAY_SIZE(ARRAY_SIZE),
        .EXT_ADDR_EN(EXT_ADDR_EN),
        .VECTOR_UNIT_EN(VECTOR_UNIT_EN),
        .QUANTIZER_LANES(QUANTIZER_LANES),
        .RELU_LANES(RELU_LANES),
        .QUANTIZER_PIPE_DEPTH(QUANTIZER_PIPE_DEPTH),
        .WEIGHT_OVERLAP_EN(WEIGHT_OVERLAP_EN),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .BUFFER_SIZE(BUFFER_SIZE),
        .WEIGHT_REGION_WORDS(WEIGHT_REGION_WORDS),
        .PROG_DEPTH(PROG_DEPTH)
    ) u_top (
        .clk(core_clk),
        .rst(core_reset),
        .rx(rx),
        .tx(tx),
        .led_rst(led_rst)
    );
endmodule
