// Looking back at this, this is wrong


`include "mac.sv"

module mac_array #(
	parameter ARRAY_SIZE     	 = 2,
	parameter COMPUTE_DATA_WIDTH 	 = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16
    ) (
	input logic clk, rst, compute, load_en, 
	input logic  [COMPUTE_DATA_WIDTH-1:0] 	  in          [ARRAY_SIZE-1:0],
	output logic [ACCUMULATOR_DATA_WIDTH-1:0] accumulator [ARRAY_SIZE-1:0]
    );

    genvar i;
    generate 
	for (i = 0; i < ARRAY_SIZE; i++) begin: gen_mac
	    mac #(
		.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
		.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	    ) u_mac (
		.clk(clk),
		.rst(rst),
		.compute(compute),
		.load_en(load_en),
		.in(in[i]),
		.accumulator(accumulator[i])
	    );
	end
    endgenerate

endmodule: mac_array
