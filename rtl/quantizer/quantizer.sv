
`timescale 1ns/1ps

module quantizer #(
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4
    ) ( 
	input  signed logic [ACCUMULATOR_DATA_WIDTH-1:0]  in,
	output signed logic [COMPUTE_DATA_WIDTH-1:0] 	  result
    );
    
    assign result = in >>> (ACCUMULATOR_DATA_WIDTH - COMPUTE_DATA_WIDTH);

endmodule: quantizer
