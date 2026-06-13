
`timescale 1ns/1ps

module quantizer_array #(
	parameter QUANTIZER_SIZE         = 8*8,
	parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE),
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4
    ) (
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] ins     [QUANTIZER_SIZE-1:0],
	input  logic                                      requant_enable,
	input  logic [15:0]                               requant_multiplier,
	input  logic [15:0]                               requant_right_shift,
	output logic signed [COMPUTE_DATA_WIDTH-1:0]     results [QUANTIZER_SIZE-1:0]
    );

    genvar i;
    generate 
	for (i = 0; i < QUANTIZER_SIZE; i++) begin: create_array
	    quantizer #(
		.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
		.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
	    ) u_quant (
		.in(ins[i]),
		.requant_enable(requant_enable),
		.requant_multiplier(requant_multiplier),
		.requant_right_shift(requant_right_shift),
		.result(results[i])
	    );
	end
    endgenerate

endmodule: quantizer_array
