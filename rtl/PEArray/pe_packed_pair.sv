/*
 * Packed signed-INT8 MAC pair: two MACs sharing operand c via one wide multiply.
 * Forms (a<<<18 + b)*c, extracts lanes for a*c and b*c, accumulates separately.
 * Bit-exact to two independent reference MACs when lane extraction applies the
 * WP487 signed correction (+1 to the high lane when the low lane is negative).
 */

`timescale 1ns/1ps

module pe_packed_pair #(
    parameter COMPUTE_DATA_WIDTH     = 8,
    parameter ACCUMULATOR_DATA_WIDTH = 32,
    parameter PACK_SHIFT             = 18
) (
    input  logic clk, rst, compute, load_en,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]      c_in,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]      a_in,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]      b_in,
    input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_a_in,
    input  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_b_in,
    output logic signed [COMPUTE_DATA_WIDTH-1:0]      c_out,
    output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_a_out,
    output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_b_out
);

    localparam int PACKED_OPERAND_WIDTH = PACK_SHIFT + COMPUTE_DATA_WIDTH + 1;
    localparam int PACKED_PRODUCT_WIDTH = PACKED_OPERAND_WIDTH + COMPUTE_DATA_WIDTH;
    // Signed product lane width (INT8xINT8 -> 16, INT4xINT4 -> 8).
    localparam int LANE_PROD_WIDTH = 2 * COMPUTE_DATA_WIDTH;

    logic signed [COMPUTE_DATA_WIDTH-1:0] a_reg;
    logic signed [COMPUTE_DATA_WIDTH-1:0] b_reg;

    logic signed [PACKED_OPERAND_WIDTH-1:0] packed_operand;
    logic signed [COMPUTE_DATA_WIDTH-1:0] c_mul;
    // Keep the multiply as a standalone USE_DSP net so Vivado cannot silently
    // absorb a small INT4 product into LUTs when measuring DSP-per-pair.
    (* use_dsp = "yes" *) logic signed [PACKED_PRODUCT_WIDTH-1:0] packed_prod;
    assign packed_prod = $signed(packed_operand) * $signed(c_mul);

    logic signed [LANE_PROD_WIDTH-1:0] lane_b_c;
    logic signed [LANE_PROD_WIDTH-1:0] lane_a_c_raw;
    logic signed [LANE_PROD_WIDTH-1:0] lane_a_c;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] mac_a;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] mac_b;

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] safe_s8(
        input logic signed [COMPUTE_DATA_WIDTH-1:0] value
    );
`ifdef ICARUS
        safe_s8 = (^value === 1'bx) ? '0 : value;
`else
        safe_s8 = value;
`endif
    endfunction

    function automatic logic signed [ACCUMULATOR_DATA_WIDTH-1:0] safe_acc(
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] value
    );
`ifdef ICARUS
        safe_acc = (^value === 1'bx) ? '0 : value;
`else
        safe_acc = value;
`endif
    endfunction

    always_comb begin
        logic signed [COMPUTE_DATA_WIDTH-1:0] a_safe;
        logic signed [COMPUTE_DATA_WIDTH-1:0] b_safe;
        logic signed [COMPUTE_DATA_WIDTH-1:0] c_safe;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] psum_a_safe;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] psum_b_safe;

        a_safe      = safe_s8(a_reg);
        b_safe      = safe_s8(b_reg);
        c_safe      = safe_s8(c_in);
        psum_a_safe = safe_acc(partial_sum_a_in);
        psum_b_safe = safe_acc(partial_sum_b_in);

        packed_operand = ($signed(a_safe) <<< PACK_SHIFT) + $signed(b_safe);
        c_mul = c_safe;

        lane_b_c = $signed(packed_prod[LANE_PROD_WIDTH-1:0]);
        lane_a_c_raw = $signed(packed_prod[PACK_SHIFT +: LANE_PROD_WIDTH]);
        if (lane_b_c < 0)
            lane_a_c = lane_a_c_raw + LANE_PROD_WIDTH'(1);
        else
            lane_a_c = lane_a_c_raw;

        mac_a = psum_a_safe + {{(ACCUMULATOR_DATA_WIDTH-LANE_PROD_WIDTH){lane_a_c[LANE_PROD_WIDTH-1]}}, lane_a_c};
        mac_b = psum_b_safe + {{(ACCUMULATOR_DATA_WIDTH-LANE_PROD_WIDTH){lane_b_c[LANE_PROD_WIDTH-1]}}, lane_b_c};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            a_reg             <= '0;
            b_reg             <= '0;
            c_out             <= '0;
            partial_sum_a_out <= '0;
            partial_sum_b_out <= '0;
        end else begin
            c_out <= safe_s8(c_in);
            if (load_en) begin
                a_reg <= safe_s8(a_in);
                b_reg <= safe_s8(b_in);
            end else if (compute) begin
                partial_sum_a_out <= mac_a;
                partial_sum_b_out <= mac_b;
            end
        end
    end

endmodule: pe_packed_pair
