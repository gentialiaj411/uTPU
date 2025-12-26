`include "pe.sv"

module pe_array #(
	parameter ARRAY_SIZE = 2,
	parameter ARRAY_SIZE_WIDTH = $clog2(ARRAY_SIZE),
	parameter COMPUTE_DATA_WIDTH = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16
    ) (
	input logic clk, rst, compute, load_en,
	input logic [COMPUTE_DATA_WIDTH-1:0] ins [ARRAY_SIZE_WIDTH-1:0],
	output logic [ACCUMULATOR_DATA_WIDTH-1:0] results [ARRAY_SIZE_WIDTH-1:0]
    );

    
    logic [ACCUMULATOR_DATA_WIDTH-1:0] accumulators [ARRAY_SIZE_WIDTH-1:0][ARRAY_SIZE_WIDTH-1:0];
    logic [COMPUTE_DATA_WIDTH-1:0] activations [ARRAY_SIZE_WIDTH-1:0][ARRAY_SIZE_WIDTH:0];

    genvar i, j;
    generate 
	for (i = 0; i < ARRAY_SIZE; i++) begin: connect_ins
	    assign activations[i][0] = ins[i];
	end

	for (i = 0; i < ARRAY_SIZE; i++) begin: connect_results
	    assign results[i] = accumulators[ARRAY_SIZE-1][i];
	end

	for (i = 0; i < ARRAY_SIZE; i++) begin: rows
	    for (j = 0; j < ARRAY_SIZE; j++) begin: cols
		pe #(
		    COMPUTE_DATA_WIDTH,
		    ACCUMULATOR_DATA_WIDTH
		) u_pe (
		    .clk(clk),
		    .rst(rst),
		    .compute(compute),
		    .load_en(load_en),
		    .in(activations[i][j]),
		    .partial_sum_in((i==0) ? '0 : accumulators[i-1][j]),
		    .activation_out(activations[i][j+1]),
		    .accumulator(accumulators[i][j])
		);
	    end
	end
    endgenerate
endmodule: pe_array
	
