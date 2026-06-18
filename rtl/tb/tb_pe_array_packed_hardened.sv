`timescale 1ns/1ps

// Isolated hardened shape matrix for pe_array_packed vs baseline pe_array.
// Does not replace tb_pe_array_packed.sv / the 402-GEMM green target.

module packed_hardened_shape_harness #(
    parameter int ARRAY_SIZE             = 8,
    parameter int COMPUTE_DATA_WIDTH     = 8,
    parameter int ACCUMULATOR_DATA_WIDTH = 32
) (
    input logic clk,
    output int  total_cases,
    output int  fail_cases
);

    localparam int LANES = ARRAY_SIZE * ARRAY_SIZE;

    logic rst;
    logic compute;
    logic load_en;

    logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations_flat [LANES-1:0];

    int cases_run;
    int cases_failed;

    int first_mismatch_row;
    int first_mismatch_col;
    int first_mismatch_cycle;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] first_mismatch_base;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] first_mismatch_pack;
    bit first_mismatch_valid;

    pe_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) u_baseline (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .datas_in(datas_in),
        .weights_in(weights_in),
        .results()
    );

    pe_array_packed #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) u_packed (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .datas_in(datas_in),
        .weights_in(weights_in),
        .results()
    );

    function automatic logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_bottom(input int mesh_col);
        if (mesh_col % 2 == 0)
            packed_bottom = u_packed.accum_a[ARRAY_SIZE-1][mesh_col/2];
        else
            packed_bottom = u_packed.accum_b[ARRAY_SIZE-1][mesh_col/2];
    endfunction

    task automatic clear_first_mismatch();
        first_mismatch_valid = 1'b0;
        first_mismatch_row = -1;
        first_mismatch_col = -1;
        first_mismatch_cycle = -1;
        first_mismatch_base = '0;
        first_mismatch_pack = '0;
    endtask

    task automatic note_mismatch(
        input int row,
        input int col,
        input int cycle,
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_val,
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_val
    );
        if (!first_mismatch_valid) begin
            first_mismatch_valid = 1'b1;
            first_mismatch_row = row;
            first_mismatch_col = col;
            first_mismatch_cycle = cycle;
            first_mismatch_base = base_val;
            first_mismatch_pack = pack_val;
        end
    endtask

    task automatic apply_reset();
        rst = 1'b1;
        compute = 1'b0;
        load_en = 1'b0;
        foreach (datas_in[i])
            datas_in[i] = '0;
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;
        @(posedge clk);
    endtask

    task automatic load_weights();
        load_en = 1'b1;
        compute = 1'b0;
        @(posedge clk);
        load_en = 1'b0;
        @(posedge clk);
    endtask

    task automatic clear_stimulus();
        int i;
        begin
            for (i = 0; i < LANES; i++) begin
                weights_in[i] = '0;
                activations_flat[i] = '0;
            end
        end
    endtask

    task automatic fill_case(
        input int seed,
        input int logical_m,
        input int logical_k,
        input int logical_n,
        input bit use_corners,
        input bit odd_n_zero_last_lane
    );
        int r;
        int c;
        int k;
        int state;
        logic signed [COMPUTE_DATA_WIDTH-1:0] corners [0:3];
        begin
            corners[0] = -8'sd128;
            corners[1] = -8'sd1;
            corners[2] = 8'sd0;
            corners[3] = 8'sd127;
            clear_stimulus();
            state = seed;
            for (r = 0; r < logical_m; r++) begin
                for (k = 0; k < logical_k; k++) begin
                    state = (state * 32'd1664525 + 32'd1013904223);
                    weights_in[(r * ARRAY_SIZE) + k] = state[COMPUTE_DATA_WIDTH-1:0];
                end
            end
            for (c = 0; c < logical_n; c++) begin
                for (k = 0; k < logical_k; k++) begin
                    state = (state * 32'd1664525 + 32'd1013904223);
                    activations_flat[(c * ARRAY_SIZE) + k] = state[COMPUTE_DATA_WIDTH-1:0];
                end
            end
            if (odd_n_zero_last_lane && (logical_n % 2 == 1) && (logical_n < ARRAY_SIZE)) begin
                for (r = 0; r < logical_m; r++)
                    weights_in[(r * ARRAY_SIZE) + logical_n] = '0;
            end
            if (use_corners) begin
                for (k = 0; k < 4; k++) begin
                    weights_in[(k * ARRAY_SIZE + k) % LANES] = corners[k];
                    activations_flat[(k * ARRAY_SIZE + k) % LANES] = corners[k];
                end
                weights_in[LANES-1] = -8'sd128;
                activations_flat[LANES-1] = -8'sd128;
            end
        end
    endtask

    task automatic run_stream_and_compare(
        input int batch_count,
        input int logical_m,
        input int logical_n
    );
        int cycle_count;
        int capture_cycle;
        int done_cycle;
        int row;
        int col;
        int idx;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_val;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_val;
        begin
            done_cycle = (ARRAY_SIZE * 2) + batch_count - 1;
            compute = 1'b1;
            for (cycle_count = 0; cycle_count < done_cycle; cycle_count++) begin
                capture_cycle = cycle_count + 1;
                for (row = 0; row < ARRAY_SIZE; row++) begin
                    if ((cycle_count < batch_count + row) && (cycle_count >= row)) begin
                        idx = (ARRAY_SIZE * (cycle_count - row)) + row;
                        datas_in[row] = activations_flat[idx];
                    end else begin
                        datas_in[row] = '0;
                    end
                end
                @(posedge clk);
                for (col = 0; col < logical_n; col++) begin
                    for (row = 0; row < logical_m; row++) begin
                        if ((ARRAY_SIZE + 1 + row + col) == capture_cycle) begin
                            base_val = u_baseline.accumulators[ARRAY_SIZE-1][row];
                            pack_val = packed_bottom(row);
                            if (base_val !== pack_val)
                                note_mismatch(row, col, capture_cycle, base_val, pack_val);
                        end
                    end
                end
            end
            compute = 1'b0;
            foreach (datas_in[row])
                datas_in[row] = '0;
            @(posedge clk);
        end
    endtask

    task automatic run_shape_case(
        input string class_name,
        input int seed,
        input int logical_m,
        input int logical_k,
        input int logical_n,
        input int batch_count,
        input bit use_corners,
        input bit odd_n_zero_last_lane
    );
        begin
            clear_first_mismatch();
            fill_case(seed, logical_m, logical_k, logical_n, use_corners, odd_n_zero_last_lane);
            apply_reset();
            load_weights();
            run_stream_and_compare(batch_count, logical_m, logical_n);
            cases_run++;
            if (first_mismatch_valid) begin
                cases_failed++;
                $display("HARDENED_CLASS %0s CONSTRAINT row=%0d col=%0d cycle=%0d base=%0d pack=%0d",
                         class_name,
                         first_mismatch_row,
                         first_mismatch_col,
                         first_mismatch_cycle,
                         first_mismatch_base,
                         first_mismatch_pack);
                $display("HARDENED_CAUSE %0s classification=pending root_cause=unclassified",
                         class_name);
            end else begin
                $display("HARDENED_CLASS %0s PASS", class_name);
            end
        end
    endtask

    event run_evt;

    initial begin
        @(run_evt);
        cases_run = 0;
        cases_failed = 0;

        if (ARRAY_SIZE == 8) begin
            run_shape_case("odd_n_8", 32'h0DD1_0008, 8, 8, 7, 7, 1'b1, 1'b1);
            run_shape_case("rect_k_8", 32'h0DD2_0008, 8, 4, 8, 8, 1'b0, 1'b0);
            run_shape_case("rect_mkn_8", 32'h0DD3_0008, 8, 4, 4, 4, 1'b0, 1'b0);
            run_shape_case("batch2_8", 32'h0DD4_0008, 8, 8, 2, 2, 1'b0, 1'b0);
            run_shape_case("batch4_8", 32'h0DD5_0008, 8, 8, 4, 4, 1'b0, 1'b0);
        end else if (ARRAY_SIZE == 16) begin
            run_shape_case("odd_n_16", 32'h0DD1_0010, 16, 16, 15, 15, 1'b1, 1'b1);
            run_shape_case("rect_k_16", 32'h0DD2_0010, 16, 8, 16, 16, 1'b0, 1'b0);
            run_shape_case("rect_mkn_16", 32'h0DD3_0010, 16, 8, 8, 8, 1'b0, 1'b0);
        end else if (ARRAY_SIZE == 32) begin
            run_shape_case("tile32_corner", 32'h0DD1_0020, 32, 32, 32, 32, 1'b1, 1'b0);
            run_shape_case("tile32_rand_00", 32'hA320_0000, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_01", 32'hA320_0001, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_02", 32'hA320_0002, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_03", 32'hA320_0003, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_04", 32'hA320_0004, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_05", 32'hA320_0005, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_06", 32'hA320_0006, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_07", 32'hA320_0007, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_08", 32'hA320_0008, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_09", 32'hA320_0009, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_10", 32'hA320_000A, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_11", 32'hA320_000B, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_12", 32'hA320_000C, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_13", 32'hA320_000D, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_14", 32'hA320_000E, 32, 32, 32, 32, 1'b0, 1'b0);
            run_shape_case("tile32_rand_15", 32'hA320_000F, 32, 32, 32, 32, 1'b0, 1'b0);
        end
    end

    assign total_cases = cases_run;
    assign fail_cases  = cases_failed;

endmodule

module tb_pe_array_packed_hardened;

    localparam int EXPECTED_8  = 5;
    localparam int EXPECTED_16 = 3;
    localparam int EXPECTED_32 = 17;

    logic clk;
    int total_cases;
    int fail_cases;
    int total_8;
    int fail_8;
    int total_16;
    int fail_16;
    int total_32;
    int fail_32;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    packed_hardened_shape_harness #(.ARRAY_SIZE(8))  h8  (.clk(clk), .total_cases(total_8),  .fail_cases(fail_8));
    packed_hardened_shape_harness #(.ARRAY_SIZE(16)) h16 (.clk(clk), .total_cases(total_16), .fail_cases(fail_16));
    packed_hardened_shape_harness #(.ARRAY_SIZE(32)) h32 (.clk(clk), .total_cases(total_32), .fail_cases(fail_32));

    initial begin
        -> h8.run_evt;
        wait (total_8 == EXPECTED_8);
        -> h16.run_evt;
        wait (total_16 == EXPECTED_16);
        -> h32.run_evt;
        wait (total_32 == EXPECTED_32);

        total_cases = total_8 + total_16 + total_32;
        fail_cases  = fail_8 + fail_16 + fail_32;

        $display("HARDENED_SUMMARY cases=%0d fails=%0d", total_cases, fail_cases);
        if (fail_cases == 0) begin
            $display("TB_RESULT: PASS hardened_cases=%0d", total_cases);
        end else begin
            $display("TB_RESULT: CONSTRAINT hardened_cases=%0d constraints=%0d", total_cases, fail_cases);
        end
        $finish;
    end

endmodule
