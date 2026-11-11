// Board-level pad primitives for the flash interface on the Real Digital Boolean (xc7s50csga324-1, flash S25FL128S).
// The registers in front of the pads live in qspi_pad_regs (IOB = TRUE) so that they simulate with the rest of the core.
//
// The flash clock pin is the dedicated CCLK_0 pin; after configuration a design can only reach it through STARTUPE2
// (USRCCLKO, with USRCCLKTS=0 to enable the driver). The first few USRCCLKO edges after configuration are not forwarded to
// the pin; qspi_reader clocks STARTUP_PULSES pulses with CS# high first, which absorbs them (hardware-unverified assumption).
// DQ0..DQ3 and CS# are the bank-14 dedicated-function pins used as user I/O after configuration (pins: constraints/*.xdc).
module qspi_pads (
    input  logic       pad_sclk,
    input  logic       pad_cs_n,
    input  logic [3:0] pad_io_o,
    input  logic [3:0] pad_io_t,
    output logic [3:0] pad_io_i,
    inout  wire  [3:0] qspi_dq,
    output wire        qspi_cs_n
);
    STARTUPE2 #(.PROG_USR("FALSE"), .SIM_CCLK_FREQ(0.0)) u_startup (
        .CFGCLK(), .CFGMCLK(), .EOS(), .PREQ(),
        .CLK(1'b0), .GSR(1'b0), .GTS(1'b0), .KEYCLEARB(1'b1), .PACK(1'b0),
        .USRCCLKO(pad_sclk), .USRCCLKTS(1'b0), .USRDONEO(1'b1), .USRDONETS(1'b1));
    for (genvar i = 0; i < 4; i++) begin : g_dq
        IOBUF u_iobuf (.O(pad_io_i[i]), .IO(qspi_dq[i]), .I(pad_io_o[i]), .T(pad_io_t[i]));
    end
    OBUF u_cs (.I(pad_cs_n), .O(qspi_cs_n));
endmodule
