// Real Digital Boolean (xc7s50csga324-1) top: 100 MHz oscillator -> MMCM -> 50 MHz core clock, reset, QSPI pads
// (STARTUPE2 for the flash clock), UART host link, top_decode (stories15M configuration) reading its weights
// from the on-board S25FL128S through qspi_reader. Pin names match Real Digital's master constraints file
// (clk, UART_rxd, UART_txd); the QSPI pins are in constraints/boolean_xc7s50.xdc. UART protocol: board_ctrl.sv.
module boolean_top (
    input  wire       clk,            // 100 MHz oscillator, pin F14
    input  wire       UART_rxd,       // host -> FPGA (V12)
    output wire       UART_txd,       // FPGA -> host (U11)
    inout  wire [3:0] qspi_dq,
    output wire       qspi_cs_n
);
    // ---- clock: 100 MHz * 10 = 1000 MHz VCO, / 20 = 50 MHz ----
    wire clk_fb, clk_fb_b, clk50_raw, clk50, locked;
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"), .CLKIN1_PERIOD(10.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(10.0),
        .CLKOUT0_DIVIDE_F(20.0), .STARTUP_WAIT("FALSE")
    ) u_mmcm (
        .CLKIN1(clk), .CLKFBIN(clk_fb_b), .CLKFBOUT(clk_fb), .CLKOUT0(clk50_raw), .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6());
    BUFG u_fb (.I(clk_fb), .O(clk_fb_b));
    BUFG u_clk (.I(clk50_raw), .O(clk50));

    // ---- reset: held until the MMCM is locked, then 16 more cycles ----
    logic [1:0] lk_sync = 2'b00;
    logic [4:0] rst_cnt = 5'd0;
    logic       rst = 1'b1;
    always_ff @(posedge clk50) begin
        lk_sync <= {lk_sync[0], locked};
        if (!lk_sync[1]) begin rst_cnt <= '0; rst <= 1'b1; end
        else if (rst_cnt != 5'd16) rst_cnt <= rst_cnt + 1'b1;
        else rst <= 1'b0;
    end

    // ---- core + pads ----
    logic pad_sclk, pad_cs_n; logic [3:0] pad_io_o, pad_io_t, pad_io_i;
    boolean_core u_core (.clk(clk50), .rst, .uart_rx(UART_rxd), .uart_tx(UART_txd), .pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i);
    qspi_pads u_pads (.pad_sclk, .pad_cs_n, .pad_io_o, .pad_io_t, .pad_io_i, .qspi_dq, .qspi_cs_n);
endmodule
