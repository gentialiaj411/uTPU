/*
*
*  PE/MXU ARRAY Module:
*  	This is the array of mxu units. Inputs are fed from the right from
*  	matrix A, weights are loaded coming from matrix B, and the
*  	partial_sums--the last of which is the result--flow downward.		
*
* 	Currently, the size of the array is 8x8 controlled by ARRAY_SIZE.
*
*	
*
*
*/

`timescale 1ns/1ps

module pe_array #(
	parameter ARRAY_SIZE 		 = 8,
	parameter ARRAY_SIZE_WIDTH 	 = $clog2(ARRAY_SIZE),
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter BUFFER_WORD_SIZE       = 16,
	parameter NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH,
	parameter WEIGHT_OVERLAP_EN      = 0
    ) (
	input  logic clk, rst, compute, load_en, weight_commit,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0],
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0],
	output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results    [ARRAY_SIZE-1:0]
    );
    
     
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] accumulators [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations  [ARRAY_SIZE-1:0][ARRAY_SIZE:0];
     

    genvar i, j;
    generate 
	for (i = 0; i < ARRAY_SIZE; i++) begin: connect_ins
	    assign activations[i][0] = datas_in[i];
	end

	for (i = 0; i < ARRAY_SIZE; i++) begin: connect_results
	    assign results[i] = accumulators[ARRAY_SIZE-1][i];
	end

	for (i = 0; i < ARRAY_SIZE; i++) begin: gen_rows
	    for (j = 0; j < ARRAY_SIZE; j++) begin: gen_cols
                if (i == 0) begin: gen_top_row
		    pe #(
		        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
		        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
		        .WEIGHT_OVERLAP_EN(WEIGHT_OVERLAP_EN)
		    ) u_pe (
		        .clk(clk),
		        .rst(rst),
		        .compute(compute),
		        .load_en(load_en),
		        .weight_commit(weight_commit),
		        .data_in(activations[i][j]),
		        .weight_in(weights_in[i*ARRAY_SIZE+j]),
		        .partial_sum_in({ACCUMULATOR_DATA_WIDTH{1'b0}}),
		        .data_out(activations[i][j+1]),
		        .partial_sum_out(accumulators[i][j])
		    );
                end else begin: gen_non_top_row
		    pe #(
		        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
		        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
		        .WEIGHT_OVERLAP_EN(WEIGHT_OVERLAP_EN)
		    ) u_pe (
		        .clk(clk),
		        .rst(rst),
		        .compute(compute),
		        .load_en(load_en),
		        .weight_commit(weight_commit),
		        .data_in(activations[i][j]),
		        .weight_in(weights_in[i*ARRAY_SIZE+j]),
		        .partial_sum_in(accumulators[i-1][j]),
		        .data_out(activations[i][j+1]),
		        .partial_sum_out(accumulators[i][j])
		    );
                end
	    end
	end
    endgenerate
endmodule: pe_array
	
