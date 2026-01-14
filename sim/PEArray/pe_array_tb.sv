`timescale 1ns/1ps

module pe_array_tb;
    
    localparam int ARRAY_SIZE = 2;
    localparam int DATA_WIDTH = 4;
    localparam int ACCUMULATOR_DATA_WIDTH = 16;
    localparam int COMPUTE_DATA_WIDTH = 4;

    logic clk, rst, compute, load_en;

    logic signed [DATA_WIDTH-1:0]             datas_in   [ARRAY_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0]             weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results    [ARRAY_SIZE-1:0];


    pe_array #(
	.ARRAY_SIZE(ARRAY_SIZE),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) dut (
	.clk(clk),
	.rst(rst),
	.compute(compute),
	.load_en(load_en),
	.datas_in(datas_in),
	.weights_in(weights_in),
	.results(results)
    );

    //clk
    initial clk = 0;
    always #5 clk = ~clk;


    // Helper task to clear weights
    task automatic clear_weights();
	for (int i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) begin
	    weights_in[i] = '0;
	end
    endtask

    task automatic clear_datas();
	for (int i = 0; i < ARRAY_SIZE; i++) begin
	    datas_in[i] = '0;
	end
    endtask

    initial begin 
	$monitor("results[0]=%0d results[1]=%0d", results[0], results[1]);
	rst = 1;
	compute = 0;
	clear_datas();
	clear_weights();

	repeat (3) @(posedge clk);
	rst = 0;
	
	//LOAD 
	// B = [[5,6],[7,8]]
	
	load_en = 1;
	compute = 0;

	//inject all weights
	weights_in[0] = 1;
	weights_in[1] = 1;
	weights_in[2] = 1;
	weights_in[3] = 1;
	@(posedge clk);

	load_en = 0;

	clear_weights();

	//COMPUTE 
	// A = [[1,2],[3,4]]
	
	compute = 1;

	//cycle 0
	// put 1 on first row
	datas_in[0] = 1;
	@(posedge clk);

	//cycle 1
	datas_in[0] = 1;
	datas_in[1] = 1;
	@(posedge clk);

	//cycle 2
	datas_in[0] = 0;
	datas_in[1] = 1;
	@(posedge clk);

	clear_datas();
	clear_weights();
	repeat (6) @(posedge clk)

	compute = 0;
	
	$display("results[0]=%0d results[1]=%0d", results[0], results[1]);
	$finish;
    end
endmodule
	

