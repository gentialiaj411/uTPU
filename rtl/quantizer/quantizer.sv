
`timescale 1ns/1ps

module quantizer #(
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4
    ) ( 
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0]  in,
	input  logic                                      requant_enable,
	input  logic [15:0]                               requant_multiplier,
	input  logic [15:0]                               requant_right_shift,
	output logic signed [COMPUTE_DATA_WIDTH-1:0] 	  result
    );

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] quantize_impl(
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x,
        input logic use_requant,
        input logic [15:0] multiplier,
        input logic [15:0] right_shift
    );
        logic signed [ACCUMULATOR_DATA_WIDTH+16-1:0] product;
        logic signed [ACCUMULATOR_DATA_WIDTH+16-1:0] shifted;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] legacy_shifted;
        logic signed [ACCUMULATOR_DATA_WIDTH+16-1:0] hi;
        logic signed [ACCUMULATOR_DATA_WIDTH+16-1:0] lo;
        begin
            legacy_shifted = x >>> (ACCUMULATOR_DATA_WIDTH - COMPUTE_DATA_WIDTH);
            product = $signed(x) * $signed({1'b0, multiplier});
            shifted = (right_shift != 16'd0) ? (product >>> right_shift) : product;
            hi = ($signed(1) <<< (COMPUTE_DATA_WIDTH-1)) - 1;
            lo = -($signed(1) <<< (COMPUTE_DATA_WIDTH-1));
            if (!use_requant)
                quantize_impl = legacy_shifted[COMPUTE_DATA_WIDTH-1:0];
            else if (shifted > hi)
                quantize_impl = hi[COMPUTE_DATA_WIDTH-1:0];
            else if (shifted < lo)
                quantize_impl = lo[COMPUTE_DATA_WIDTH-1:0];
            else
                quantize_impl = shifted[COMPUTE_DATA_WIDTH-1:0];
        end
    endfunction

    assign result = quantize_impl(in, requant_enable, requant_multiplier, requant_right_shift);

endmodule: quantizer
