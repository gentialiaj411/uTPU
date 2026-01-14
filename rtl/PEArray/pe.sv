/*
*  PE/MXU Module: 
*      These are the actual compute units inside the systolic array. They have
*      inputs of data which are from matrix A, weight which are from matrix B,
*      and partial_sum which is from the unit above. If compute enabled, it
*      adds the partial sum it is given to the product of the weight stored
*      inside and the data passing through it. It then outputs the data to the
*      unit to the left, the sum is passed downward. To load the weights from
*      B, load_en must be on. 
* 
*
*/

`timescale 1ns/1ps


module pe #(
	parameter COMPUTE_DATA_WIDTH = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16
    ) (
	input  logic clk, rst, compute, load_en,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     data_in,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weight_in,
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_in,
	output logic signed [COMPUTE_DATA_WIDTH-1:0]     data_out,
	output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_out
    );

    logic signed [COMPUTE_DATA_WIDTH-1:0] weight;

    always_ff @(posedge clk) begin
        if (rst) begin
	    weight          <= '0;
            data_out        <= '0;
            partial_sum_out <= '0;
        end else begin
            data_out <= data_in;
            if (load_en) 
		weight <= weight_in;
            else if (compute) 
		partial_sum_out <= partial_sum_in + 
		               $signed((ACCUMULATOR_DATA_WIDTH)'($signed(data_in) * $signed(weight)));
        end
    end
endmodule: pe

