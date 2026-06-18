/*
 * Packed-DSP systolic array: pairs adjacent weight columns that share the
 * streamed activation at even mesh columns. Each packed cell performs two
 * MACs via (w1<<<18 + w2)*activation with skewed low-lane extraction (+1
 * high-lane correction, low lane taken from the previous cycle product).
 * Odd mesh columns are single-cycle activation passthrough registers so the
 * horizontal wavefront matches pe_array.sv timing.
 */

`timescale 1ns/1ps

module pe_packed_skewed #(
    parameter COMPUTE_DATA_WIDTH     = 8,
    parameter ACCUMULATOR_DATA_WIDTH = 32,
    parameter PACK_SHIFT             = 18,
    parameter PACKED_PRODUCT_WIDTH   = 48
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

    logic signed [COMPUTE_DATA_WIDTH-1:0] a_reg;
    logic signed [COMPUTE_DATA_WIDTH-1:0] b_reg;

    logic signed [PACKED_OPERAND_WIDTH-1:0] packed_operand;
    (* use_dsp = "yes" *) logic signed [PACKED_PRODUCT_WIDTH-1:0] packed_prod;
    logic signed [PACKED_PRODUCT_WIDTH-1:0] packed_prod_prev;

    logic signed [15:0] lane_a_c;
    logic signed [15:0] lane_b_c_prev;
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

    function automatic logic signed [15:0] extract_high_lane(
        input logic signed [PACKED_PRODUCT_WIDTH-1:0] prod
    );
        logic signed [15:0] lane_b;
        logic signed [15:0] lane_a_raw;
        begin
            lane_b     = $signed(prod[15:0]);
            lane_a_raw = $signed(prod[33:18]);
            if (lane_b < 0)
                extract_high_lane = lane_a_raw + 16'sd1;
            else
                extract_high_lane = lane_a_raw;
        end
    endfunction

    function automatic logic signed [15:0] extract_low_lane(
        input logic signed [PACKED_PRODUCT_WIDTH-1:0] prod
    );
        extract_low_lane = $signed(prod[15:0]);
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
        packed_prod    = $signed(packed_operand) * $signed(c_safe);

        lane_a_c      = extract_high_lane(packed_prod);
        lane_b_c_prev = extract_low_lane(packed_prod_prev);

        mac_a = psum_a_safe + {{(ACCUMULATOR_DATA_WIDTH-16){lane_a_c[15]}}, lane_a_c};
        mac_b = psum_b_safe + {{(ACCUMULATOR_DATA_WIDTH-16){lane_b_c_prev[15]}}, lane_b_c_prev};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            a_reg             <= '0;
            b_reg             <= '0;
            c_out             <= '0;
            packed_prod_prev  <= '0;
            partial_sum_a_out <= '0;
            partial_sum_b_out <= '0;
        end else begin
            c_out <= safe_s8(c_in);
            if (load_en) begin
                a_reg <= safe_s8(a_in);
                b_reg <= safe_s8(b_in);
            end else if (compute) begin
                packed_prod_prev  <= packed_prod;
                partial_sum_a_out <= mac_a;
                partial_sum_b_out <= mac_b;
            end
        end
    end

endmodule: pe_packed_skewed

module pe_act_delay #(
    parameter COMPUTE_DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0] data_in,
    output logic signed [COMPUTE_DATA_WIDTH-1:0] data_out
);

    always_ff @(posedge clk) begin
        if (rst)
            data_out <= '0;
        else
            data_out <= data_in;
    end

endmodule: pe_act_delay

module pe_array_packed #(
    parameter ARRAY_SIZE             = 8,
    parameter ARRAY_SIZE_WIDTH       = $clog2(ARRAY_SIZE),
    parameter COMPUTE_DATA_WIDTH     = 8,
    parameter ACCUMULATOR_DATA_WIDTH   = 32,
    parameter BUFFER_WORD_SIZE       = 16,
    parameter NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH
) (
    input  logic clk, rst, compute, load_en,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0],
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0],
    output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results    [ARRAY_SIZE-1:0]
);

    generate
        if ((ARRAY_SIZE % 2) != 0) begin: gen_even_array_size_guard
            initial begin
                $error("pe_array_packed requires even ARRAY_SIZE, got %0d", ARRAY_SIZE);
                $fatal(1, "invalid ARRAY_SIZE for pe_array_packed");
            end
        end
    endgenerate

    localparam int PHYS_COLS = ARRAY_SIZE / 2;

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] accum_a [ARRAY_SIZE-1:0][PHYS_COLS-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] accum_b [ARRAY_SIZE-1:0][PHYS_COLS-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations [ARRAY_SIZE-1:0][ARRAY_SIZE:0];

    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin: connect_ins
            assign activations[i][0] = datas_in[i];
        end

        for (j = 0; j < ARRAY_SIZE; j++) begin: connect_results
            if (j % 2 == 0) begin: map_even_col
                assign results[j] = accum_a[ARRAY_SIZE-1][j/2];
            end else begin: map_odd_col
                assign results[j] = accum_b[ARRAY_SIZE-1][j/2];
            end
        end

        for (i = 0; i < ARRAY_SIZE; i++) begin: gen_rows
            for (j = 0; j < ARRAY_SIZE; j = j + 2) begin: gen_packed_cols
                if (i == 0) begin: gen_top_packed
                    pe_packed_skewed #(
                        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
                        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
                    ) u_packed (
                        .clk(clk),
                        .rst(rst),
                        .compute(compute),
                        .load_en(load_en),
                        .c_in(activations[i][j]),
                        .a_in(weights_in[i*ARRAY_SIZE + j]),
                        .b_in(weights_in[i*ARRAY_SIZE + j + 1]),
                        .partial_sum_a_in({ACCUMULATOR_DATA_WIDTH{1'b0}}),
                        .partial_sum_b_in({ACCUMULATOR_DATA_WIDTH{1'b0}}),
                        .c_out(activations[i][j+1]),
                        .partial_sum_a_out(accum_a[i][j/2]),
                        .partial_sum_b_out(accum_b[i][j/2])
                    );
                end else begin: gen_row_packed
                    pe_packed_skewed #(
                        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
                        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
                    ) u_packed (
                        .clk(clk),
                        .rst(rst),
                        .compute(compute),
                        .load_en(load_en),
                        .c_in(activations[i][j]),
                        .a_in(weights_in[i*ARRAY_SIZE + j]),
                        .b_in(weights_in[i*ARRAY_SIZE + j + 1]),
                        .partial_sum_a_in(accum_a[i-1][j/2]),
                        .partial_sum_b_in(accum_b[i-1][j/2]),
                        .c_out(activations[i][j+1]),
                        .partial_sum_a_out(accum_a[i][j/2]),
                        .partial_sum_b_out(accum_b[i][j/2])
                    );
                end
            end

            for (j = 1; j < ARRAY_SIZE; j = j + 2) begin: gen_odd_delays
                pe_act_delay #(
                    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
                ) u_act_delay (
                    .clk(clk),
                    .rst(rst),
                    .data_in(activations[i][j]),
                    .data_out(activations[i][j+1])
                );
            end
        end
    endgenerate

endmodule: pe_array_packed
