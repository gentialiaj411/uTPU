`timescale 1ns/1ps

module tb_pe_array_packed_odd_guard;
    logic clk = 1'b0;
    logic rst = 1'b0;
    logic compute = 1'b0;
    logic load_en = 1'b0;
    logic signed [7:0] datas_in [6:0];
    logic signed [7:0] weights_in [48:0];
    logic signed [31:0] results [6:0];

    always #5 clk = ~clk;

    pe_array_packed #(
        .ARRAY_SIZE(7),
        .COMPUTE_DATA_WIDTH(8),
        .ACCUMULATOR_DATA_WIDTH(32)
    ) dut (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .datas_in(datas_in),
        .weights_in(weights_in),
        .results(results)
    );

    initial begin
        $display("TB_RESULT: FAIL guard did not fire");
        #20;
        $fatal(1, "Odd ARRAY_SIZE guard did not trigger");
    end
endmodule
