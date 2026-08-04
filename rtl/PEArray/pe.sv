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
*      Weight double-buffer (WEIGHT_OVERLAP_EN=1): load_en writes weight_shadow
*      only; weight_commit swaps shadow into the active MAC weight. Legacy mode
*      (WEIGHT_OVERLAP_EN=0, default): load_en also writes active weight so
*      existing LOAD→COMPUTE programs stay bit-exact. Overlap mode removes
*      load_en from the partial_sum_out CE path (100 MHz binder).
*/

`timescale 1ns/1ps


module pe #(
	parameter COMPUTE_DATA_WIDTH = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter WEIGHT_OVERLAP_EN = 0
    ) (
	input  logic clk, rst, compute, load_en, weight_commit,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     data_in,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weight_in,
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_in,
	output logic signed [COMPUTE_DATA_WIDTH-1:0]     data_out,
	output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_out
    );

    logic signed [COMPUTE_DATA_WIDTH-1:0] weight;
    logic signed [COMPUTE_DATA_WIDTH-1:0] weight_shadow;
    // Encourage DSP inference for MAC to reduce LUT usage.
    (* use_dsp = "yes" *) logic signed [ACCUMULATOR_DATA_WIDTH-1:0] prod;
    (* use_dsp = "yes" *) logic signed [ACCUMULATOR_DATA_WIDTH-1:0] mac;

    always_comb begin
`ifdef ICARUS
        logic signed [COMPUTE_DATA_WIDTH-1:0] data_safe;
        logic signed [COMPUTE_DATA_WIDTH-1:0] weight_safe;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] psum_safe;
        data_safe = (^data_in === 1'bx) ? '0 : data_in;
        weight_safe = (^weight === 1'bx) ? '0 : weight;
        psum_safe = (^partial_sum_in === 1'bx) ? '0 : partial_sum_in;
        prod = $signed(data_safe) * $signed(weight_safe);
        mac  = psum_safe + prod;
`else
        prod = $signed(data_in) * $signed(weight);
        mac  = partial_sum_in + prod;
`endif
    end

    always_ff @(posedge clk) begin
        if (rst) begin
	    weight          <= '0;
            weight_shadow   <= '0;
            data_out        <= '0;
            partial_sum_out <= '0;
	end else begin
            data_out <= data_in;
	    if (load_en)
`ifdef ICARUS
                weight_shadow <= (^weight_in === 1'bx) ? '0 : weight_in;
`else
		weight_shadow <= weight_in;
`endif
            if (weight_commit)
                weight <= weight_shadow;
            else if (WEIGHT_OVERLAP_EN == 0 && load_en)
`ifdef ICARUS
                weight <= (^weight_in === 1'bx) ? '0 : weight_in;
`else
                weight <= weight_in;
`endif
	    else if (compute)
		partial_sum_out <= mac;
        end
    end
endmodule: pe
