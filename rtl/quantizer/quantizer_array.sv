`include "quantizer.sv"


module quantizer_array #(
	parameter QUANTIZER_SIZE         = 4,
	parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE),
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4
    ) (
	input  logic [ACCUMULATOR_DATA_WIDTH-1:0] ins     [QUANTIZER_SIZE-1:0],
	output logic [COMPUTE_DATA_WIDTH-1:0]     results [QUANTIZER_SIZE-1:0]
    );

    genvar i;
    generate 
	for (i = 0; i < QUANTIZER_SIZE; i++) begin: create_array
	    quantizer #(
		.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
		.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
	    ) u_quant (
		.in(ins[i]),
		.result(results[i])
	    );
	end
    endgenerate

endmodule: quantizer_array
