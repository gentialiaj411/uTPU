// Board-level pad wrapper for qspi_reader on the Real Digital Boolean (xc7s50csga324-1, flash S25FL128S).
//
// The flash clock pin is the dedicated CCLK_0 pin; after configuration a design can only reach it
// through the STARTUPE2 primitive (USRCCLKO, with USRCCLKTS=0 to enable the driver). Per the project
// notes the first three USRCCLKO edges after configuration are not forwarded to the pin; qspi_reader
// clocks STARTUP_PULSES pulses with CS# high first, which absorbs them. This is a hardware-unverified
// assumption (see docs/r3_progress.md, M4 open issues).
// DQ0..DQ3 and CS# are ordinary (dedicated-function) bank-14 pins used as user I/O after configuration.
// Pin locations are NOT set here; they must come from Real Digital's schematic / constraints file.
module qspi_pads (
    input  logic       sclk,
    input  logic       cs_n,
    input  logic [3:0] io_o,
    input  logic [3:0] io_oe,
    output logic [3:0] io_i,
    inout  wire  [3:0] qspi_dq,
    output wire        qspi_cs_n
);
    STARTUPE2 #(.PROG_USR("FALSE"), .SIM_CCLK_FREQ(0.0)) u_startup (
        .CFGCLK(), .CFGMCLK(), .EOS(), .PREQ(),
        .CLK(1'b0), .GSR(1'b0), .GTS(1'b0), .KEYCLEARB(1'b1), .PACK(1'b0),
        .USRCCLKO(sclk), .USRCCLKTS(1'b0), .USRDONEO(1'b1), .USRDONETS(1'b1));
    for (genvar i = 0; i < 4; i++) begin : g_dq
        IOBUF u_iobuf (.O(io_i[i]), .IO(qspi_dq[i]), .I(io_o[i]), .T(~io_oe[i]));
    end
    OBUF u_cs (.I(cs_n), .O(qspi_cs_n));
endmodule
