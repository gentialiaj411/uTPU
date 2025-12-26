module leaky_relu_array #(
	parameter RELU_SIZE = 4,
	parameter RELU_ADDRESS_WIDTH = $clog2(RELU_SIZE),
	parameter ALPHA = 2,
	parameter COMPUTE_DATA_WIDTH = 4
    ) ( 
	input logic signed [COMPUTE_DATA_WIDTH-1:0] in [RELU_ADDRESS_WIDTH-1:0],
	output logic signed [COMPUTE_DATA_WIDTH-1:0] result [RELU_ADDRESS_WIDTH-1:]
    );

    genvar i;
    generate
	for (i = 0; i < RELU_SIZE; i++) begin: array_gen
	    
	end
    endgenerate
endmodule: leaky_relu_array

