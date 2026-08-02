`timescale 1ns/1ps

// Requant multiply operand width matches the DSP48E1 signed A-port (25 bits).
// Saturating the accumulator into this range before the multiply lets Vivado
// infer one DSP48 per quantizer instead of two for a 32x16 product.
`ifndef REQUANT_MUL_OPERAND_WIDTH
`define REQUANT_MUL_OPERAND_WIDTH 25
`endif

// QUANTIZER_PIPE_DEPTH:
//   0 = combinational (Step1+2 default until Fmax proves pipeline worth it)
//   1 = register final result only (Step 2b)
//   3 = product / shift / clamp stages (targets DSP MREG + shorter paths)
module quantizer #(
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter REQUANT_MUL_OPERAND_WIDTH = `REQUANT_MUL_OPERAND_WIDTH,
	parameter int QUANTIZER_PIPE_DEPTH = 0
    ) ( 
	input  logic                                      clk,
	input  logic                                      rst,
	input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0]  in,
	input  logic                                      requant_enable,
	input  logic [15:0]                               requant_multiplier,
	input  logic [15:0]                               requant_right_shift,
	output logic signed [COMPUTE_DATA_WIDTH-1:0] 	  result
    );

    localparam int PROD_WIDTH = REQUANT_MUL_OPERAND_WIDTH + 16;

    function automatic logic signed [ACCUMULATOR_DATA_WIDTH-1:0] saturate_requant_mul_operand(
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x
    );
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

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] clamp_to_compute(
        input logic signed [PROD_WIDTH-1:0] x
    );
        logic signed [PROD_WIDTH-1:0] hi;
        logic signed [PROD_WIDTH-1:0] lo;
        begin
            hi = ($signed(1) <<< (COMPUTE_DATA_WIDTH-1)) - 1;
            lo = -($signed(1) <<< (COMPUTE_DATA_WIDTH-1));
            if (x > hi)
                clamp_to_compute = hi[COMPUTE_DATA_WIDTH-1:0];
            else if (x < lo)
                clamp_to_compute = lo[COMPUTE_DATA_WIDTH-1:0];
            else
                clamp_to_compute = x[COMPUTE_DATA_WIDTH-1:0];
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
        logic signed [PROD_WIDTH-1:0] product;
        logic signed [PROD_WIDTH-1:0] shifted;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] legacy_shifted;
        begin
            legacy_shifted = x >>> (ACCUMULATOR_DATA_WIDTH - COMPUTE_DATA_WIDTH);
            x_sat = saturate_requant_mul_operand(x);
            x_mul = x_sat[REQUANT_MUL_OPERAND_WIDTH-1:0];
            product = $signed(x_mul) * $signed({1'b0, multiplier});
            shifted = (right_shift != 16'd0) ? (product >>> right_shift) : product;
            if (!use_requant)
                quantize_impl = legacy_shifted[COMPUTE_DATA_WIDTH-1:0];
            else
                quantize_impl = clamp_to_compute(shifted);
        end
    endfunction

    generate
        if (QUANTIZER_PIPE_DEPTH == 0) begin : g_combo
            always_comb begin
                result = quantize_impl(in, requant_enable, requant_multiplier, requant_right_shift);
            end
        end else if (QUANTIZER_PIPE_DEPTH == 1) begin : g_pipe1
            always_ff @(posedge clk) begin
                if (rst)
                    result <= '0;
                else
                    result <= quantize_impl(in, requant_enable, requant_multiplier, requant_right_shift);
            end
        end else begin : g_pipe3
            // Stage 1: saturate + multiply (register product so Vivado can pack MREG).
            logic signed [REQUANT_MUL_OPERAND_WIDTH-1:0] x_mul;
            logic signed [ACCUMULATOR_DATA_WIDTH-1:0] x_sat;
            (* use_dsp = "yes" *) logic signed [PROD_WIDTH-1:0] product_c;
            logic signed [PROD_WIDTH-1:0] product_r;
            logic use_r;
            logic [15:0] shift_r;
            logic signed [COMPUTE_DATA_WIDTH-1:0] legacy_r;

            assign x_sat = saturate_requant_mul_operand(in);
            assign x_mul = x_sat[REQUANT_MUL_OPERAND_WIDTH-1:0];
            assign product_c = $signed(x_mul) * $signed({1'b0, requant_multiplier});

            always_ff @(posedge clk) begin
                if (rst) begin
                    product_r <= '0;
                    use_r     <= 1'b0;
                    shift_r   <= '0;
                    legacy_r  <= '0;
                end else begin
                    product_r <= product_c;
                    use_r     <= requant_enable;
                    shift_r   <= requant_right_shift;
                    legacy_r  <= (in >>> (ACCUMULATOR_DATA_WIDTH - COMPUTE_DATA_WIDTH));
                end
            end

            // Stage 2: arithmetic right shift.
            logic signed [PROD_WIDTH-1:0] shifted_r;
            logic use_r2;
            logic signed [COMPUTE_DATA_WIDTH-1:0] legacy_r2;

            always_ff @(posedge clk) begin
                if (rst) begin
                    shifted_r <= '0;
                    use_r2    <= 1'b0;
                    legacy_r2 <= '0;
                end else begin
                    shifted_r <= (shift_r != 16'd0) ? (product_r >>> shift_r) : product_r;
                    use_r2    <= use_r;
                    legacy_r2 <= legacy_r;
                end
            end

            // Stage 3: clamp / mux legacy.
            always_ff @(posedge clk) begin
                if (rst)
                    result <= '0;
                else if (!use_r2)
                    result <= legacy_r2;
                else
                    result <= clamp_to_compute(shifted_r);
            end
        end
    endgenerate

endmodule: quantizer
