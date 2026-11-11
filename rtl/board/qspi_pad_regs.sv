// Registers on every pad-facing flash signal, forced into the I/O blocks (IOB = TRUE): one output flip-flop each for
// CS# and IO[3:0] data, one for each IO output-enable (stored active-low as the IOBUF's T input so no logic sits between
// flip-flop and pad), one input flip-flop for each IO. SCLK gets one more fabric register in front of STARTUPE2.USRCCLKO so
// that SCLK and the data/CS# outputs have the same latency (the flash sees data launched on the SCLK-falling phase half a
// SCLK period before the SCLK rising edge it is sampled on). Together the pad registers add 1 core cycle on every output and
// 1 on the input data path, which is why qspi_reader is instantiated with CAP_DLY = 2.
// No vendor primitives, so it simulates in iverilog; the attributes are ignored there.
module qspi_pad_regs (
    input  logic       clk,
    input  logic       sclk, cs_n,
    input  logic [3:0] io_o, io_oe,
    output logic       pad_sclk, pad_cs_n,
    output logic [3:0] pad_io_o, pad_io_t,    // pad_io_t: 1 = pad released (input), the IOBUF T polarity
    input  logic [3:0] pad_io_i,
    output logic [3:0] io_i
);
    (* IOB = "TRUE" *) logic       cs_n_r  = 1'b1;
    (* IOB = "TRUE" *) logic [3:0] io_o_r  = 4'hF;
    (* IOB = "TRUE" *) logic [3:0] io_t_r  = 4'hF;
    (* IOB = "TRUE" *) logic [3:0] io_i_r  = 4'h0;
    logic sclk_r = 1'b0;
    always_ff @(posedge clk) begin
        sclk_r  <= sclk;
        cs_n_r  <= cs_n;
        io_o_r  <= io_o;
        io_t_r  <= ~io_oe;
        io_i_r  <= pad_io_i;
    end
    assign pad_sclk = sclk_r;
    assign pad_cs_n = cs_n_r;
    assign pad_io_o = io_o_r;
    assign pad_io_t = io_t_r;
    assign io_i     = io_i_r;
endmodule
