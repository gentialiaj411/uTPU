`timescale 1ns/1ps

module pe_array_tb;
    
    localparam int ARRAY_SIZE = 2;
    localparam int DATA_WIDTH = 4;
    localparam int ACCUMULATOR_DATA_WIDTH = 16;

    logic clk, rst, compute;

    logic signed [DATA_WIDTH-1:0]             ins        [ARRAY_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0]             weights_in [ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results    [ARRAY_SIZE-1:0];


    pe_array #(
	.ARRAY_SIZE(ARRAY_SIZE),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) dut (
	.clk(clk),
	.rst(rst),
	.compute(compute),
	.ins(ins),
	.weights_in(weights_in),
	.results(results)
    );

    //clk
    initial clk = 0;
    always #5 clk = ~clk;


    // Helper task to clear ins
    task automatic 
