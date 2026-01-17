`timescale 1ns/1ps

module leaky_relu #(
	parameter ALPHA		         = 2,
	parameter COMPUTE_DATA_WIDTH     = 4
    ) (	
	input  logic signed [COMPUTE_DATA_WIDTH-1:0] in,
	output logic signed [COMPUTE_DATA_WIDTH-1:0] result
    );
    
    always_comb begin
	if (in >= 0)
	    result = in;
	else
	    result = in >>> ALPHA;
    end

endmodule: leaky_relu
