`timescale 1ns/1ps

module pe_controller #(
	parameter ARRAY_SIZE 		 = 8,
	parameter ARRAY_SIZE_WIDTH 	 = $clog2(ARRAY_SIZE),
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter BUFFER_WORD_SIZE       = 16,
	parameter NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH
    ) (
	input  logic clk, rst, compute, load_en,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_arr   [ARRAY_SIZE*ARRAY_SIZE-1:0],
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in  [ARRAY_SIZE*ARRAY_SIZE-1:0],
	output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results_arr [ARRAY_SIZE*ARRAY_SIZE-1:0]
    );
    
    localparam CYCLE_LENGTH = ARRAY_SIZE*3-1;
    localparam OUT_OFFSET   = ARRAY_SIZE+1;

    logic        [$clog2(CYCLE_LENGTH)-1:0]   cycle_count;
    logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in [ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results  [ARRAY_SIZE-1:0];
   
    int i, j;
    int k;
    int base_idx;

    always_ff @(posedge clk) begin
	if (rst) begin
	    cycle_count <= '0;
	    for (i=0; i < ARRAY_SIZE; i++)
		datas_in[i] <= '0;
	end else begin
	    if (cycle_count == CYCLE_LENGTH) 
		cycle_count <= '0;
	    else 
		cycle_count <= cycle_count + 1'b1;

	    for (i=0; i < ARRAY_SIZE; i++) begin
		if ((cycle_count < ARRAY_SIZE + i) && (cycle_count >= i)) 
		    datas_in[i] <= datas_arr[ARRAY_SIZE*(cycle_count - i) + i];
		else 
		    datas_in[i] <= '0;
	    end
	    
	    if (cycle_count >= OUT_OFFSET && cycle_count < OUT_OFFSET + ARRAY_SIZE) begin
		k = (cycle_count - OUT_OFFSET) + 1;     // 1..ARRAY_SIZE
		base_idx = (k * (k - 1)) / 2;           // 0,1,3,6,... triangular numbers

		for (j = 0; j < k; j++) begin
		    results_arr[base_idx + j] <= results[(k - 1) - j];
		end
	    end
	end
    end


    pe_array #(
	.ARRAY_SIZE(ARRAY_SIZE),
	.ARRAY_SIZE_WIDTH(ARRAY_SIZE_WIDTH),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	.BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
	.NUM_COMPUTE_LANES(NUM_COMPUTE_LANES)
    ) u_pe_array (
	.clk(clk),
	.rst(rst),
	.compute(compute),
	.load_en(load_en),
	.datas_in(datas_in),
	.weights_in(weights_in),
	.results(results)
    );

endmodule: pe_controller
