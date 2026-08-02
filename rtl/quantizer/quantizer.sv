`timescale 1ns/1ps

// Requant multiply operand width matches the DSP48E1 signed A-port (25 bits).
// Saturating the accumulator into this range before the multiply lets Vivado
// infer one DSP48 per quantizer instead of two for a 32x16 product.
`ifndef REQUANT_MUL_OPERAND_WIDTH
`define REQUANT_MUL_OPERAND_WIDTH 25
`endif

module quantizer #(
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter REQUANT_MUL_OPERAND_WIDTH = `REQUANT_MUL_OPERAND_WIDTH
    ) ( 
	input  logic                                      clk,
	input  logic                                      rst,
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0]  in,
	input  logic                                      requant_enable,
	input  logic [15:0]                               requant_multiplier,
	input  logic [15:0]                               requant_right_shift,
	output logic signed [COMPUTE_DATA_WIDTH-1:0] 	  result
    );

    function automatic logic signed [ACCUMULATOR_DATA_WIDTH-1:0] saturate_requant_mul_operand(
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x
    );
        // Signed 25-bit range: [-(2^24), 2^24 - 1]. Kept as 32-bit intermediates
        // so Icarus and Vivado agree on the shift arithmetic.
        logic signed [31:0] sat_hi;
        logic signed [31:0] sat_lo;
        logic signed [31:0] x32;
        begin
            sat_hi = (32'sd1 <<< (REQUANT_MUL_OPERAND_WIDTH - 1)) - 32'sd1;
            sat_lo = -(32'sd1 <<< (REQUANT_MUL_OPERAND_WIDTH - 1));
            x32 = x;
            if (x32 > sat_hi)
                saturate_requant_mul_operand = sat_hi[ACCUMULATOR_DATA_WIDTH-1:0];
            else if (x32 < sat_lo)
                saturate_requant_mul_operand = sat_lo[ACCUMULATOR_DATA_WIDTH-1:0];
            else
                saturate_requant_mul_operand = x;
        end
    endfunction

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] quantize_impl(
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x,
        input logic use_requant,
        input logic [15:0] multiplier,
        input logic [15:0] right_shift
    );
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x_sat;
        logic signed [REQUANT_MUL_OPERAND_WIDTH-1:0] x_mul;
        logic signed [REQUANT_MUL_OPERAND_WIDTH+16-1:0] product;
        logic signed [REQUANT_MUL_OPERAND_WIDTH+16-1:0] shifted;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] legacy_shifted;
        logic signed [REQUANT_MUL_OPERAND_WIDTH+16-1:0] hi;
        logic signed [REQUANT_MUL_OPERAND_WIDTH+16-1:0] lo;
        begin
            legacy_shifted = x >>> (ACCUMULATOR_DATA_WIDTH - COMPUTE_DATA_WIDTH);
            // DSP48E1 A-port is signed 25-bit; saturate before multiply so a
            // single DSP covers the requant product (was a 32x16 / 2-DSP infer).
            x_sat = saturate_requant_mul_operand(x);
            x_mul = x_sat[REQUANT_MUL_OPERAND_WIDTH-1:0];
            product = $signed(x_mul) * $signed({1'b0, multiplier});
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

    // One pipeline stage after the DSP multiply/shift/sat path so the closed
    // critical path (quantizer_in_reg -> DSP -> compute_to_buffer_reg) can meet
    // tighter clocks. Finalize FSM accounts for the +1 cycle latency.
    always_ff @(posedge clk) begin
        if (rst)
            result <= '0;
        else
            result <= quantize_impl(in, requant_enable, requant_multiplier, requant_right_shift);
    end

endmodule: quantizer
