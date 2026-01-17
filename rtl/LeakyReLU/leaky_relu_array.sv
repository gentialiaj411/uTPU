`timescale 1ns/1ps

module leaky_relu_array #(
	parameter RELU_SIZE          = 64,
	parameter RELU_SIZE_WIDTH    = $clog2(RELU_SIZE),
	parameter ALPHA              = 2,
	parameter COMPUTE_DATA_WIDTH = 4
    ) ( 
	input  logic signed [COMPUTE_DATA_WIDTH-1:0] in     [RELU_SIZE-1:0],
	output logic signed [COMPUTE_DATA_WIDTH-1:0] result [RELU_SIZE-1:0]
    );

    genvar i;
    generate
	for (i = 0; i < RELU_SIZE; i++) begin: array_gen
	    leaky_relu #(
		.ALPHA(ALPHA),
		.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
	    ) u_relu (
		.in(in[i]),
		.result(result[i])
	    );
	end
    endgenerate
    
endmodule: leaky_relu_array

