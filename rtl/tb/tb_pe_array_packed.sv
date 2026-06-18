`timescale 1ns/1ps

module packed_array_gemm_harness #(
    parameter int ARRAY_SIZE             = 8,
    parameter int COMPUTE_DATA_WIDTH     = 8,
    parameter int ACCUMULATOR_DATA_WIDTH = 32
) (
    input logic clk,
    output int  mismatch_count,
    output int  gemm_count
);

    localparam int WEIGHT_LANES = ARRAY_SIZE * ARRAY_SIZE;
    localparam int INPUT_LANES  = ARRAY_SIZE * ARRAY_SIZE;
    localparam int RANDOM_GEMMS = 200;

    logic rst;
    logic compute;
    logic load_en;

    logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [WEIGHT_LANES-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_results [ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_results [ARRAY_SIZE-1:0];

    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations_flat [INPUT_LANES-1:0];

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_matrix [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_matrix  [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];

    int local_mismatches;
    int local_gemms;

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
        .results(base_results)
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
        .results(pack_results)
    );

    function automatic logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_accum(input int row, input int col);
        if (col % 2 == 0) packed_accum = u_packed.accum_a[row][col/2];
        else packed_accum = u_packed.accum_b[row][col/2];
    endfunction

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

    task automatic check_all_accumulators(input string label);
        int row;
        int col;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_val;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_val;
        begin
            for (row = 0; row < ARRAY_SIZE; row++) begin
                for (col = 0; col < ARRAY_SIZE; col++) begin
                    base_val = u_baseline.accumulators[row][col];
                    pack_val = packed_accum(row, col);
                    if (base_val !== pack_val) begin
                        local_mismatches++;
                        $display("ACC_MISMATCH [%0dx%0d %s] row=%0d col=%0d base=%0d pack=%0d",
                                 ARRAY_SIZE, ARRAY_SIZE, label, row, col, base_val, pack_val);
                    end
                end
            end
        end
    endtask

    task automatic run_streamed_gemm(input int batch_count);
        int cycle_count;
        int capture_cycle;
        int done_cycle;
        int row;
        int col;
        int idx;
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
                check_all_accumulators($sformatf("cycle%0d", capture_cycle));
                for (col = 0; col < batch_count; col++) begin
                    for (row = 0; row < ARRAY_SIZE; row++) begin
                        if ((ARRAY_SIZE + 1 + row + col) == capture_cycle) begin
                            base_matrix[row][col] = u_baseline.accumulators[ARRAY_SIZE-1][row];
                            pack_matrix[row][col]  = packed_accum(ARRAY_SIZE-1, row);
                            if (base_matrix[row][col] !== pack_matrix[row][col]) begin
                                local_mismatches++;
                                $display("CAP_MISMATCH [%0dx%0d] row=%0d col=%0d base=%0d pack=%0d",
                                         ARRAY_SIZE, ARRAY_SIZE, row, col,
                                         base_matrix[row][col], pack_matrix[row][col]);
                            end
                        end
                    end
                end
            end
            compute = 1'b0;
            foreach (datas_in[row])
                datas_in[row] = '0;
            @(posedge clk);
            local_gemms++;
        end
    endtask

    task automatic fill_random_weights_activations(input int rng_seed);
        int rng_state;
        int lane;
        begin
            rng_state = rng_seed;
            for (lane = 0; lane < WEIGHT_LANES; lane++) begin
                rng_state = (rng_state * 32'd1664525 + 32'd1013904223);
                weights_in[lane] = rng_state[COMPUTE_DATA_WIDTH-1:0];
            end
            for (lane = 0; lane < INPUT_LANES; lane++) begin
                rng_state = (rng_state * 32'd1664525 + 32'd1013904223);
                activations_flat[lane] = rng_state[COMPUTE_DATA_WIDTH-1:0];
            end
        end
    endtask

    task automatic embed_corners();
        int corner_idx;
        int pos;
        logic signed [COMPUTE_DATA_WIDTH-1:0] corners [0:3];
        begin
            corners[0] = -8'sd128;
            corners[1] = -8'sd1;
            corners[2] = 8'sd127;
            corners[3] = 8'sd0;
            for (corner_idx = 0; corner_idx < 4; corner_idx++) begin
                pos = (corner_idx * ARRAY_SIZE + corner_idx) % WEIGHT_LANES;
                weights_in[pos] = corners[corner_idx];
                pos = (corner_idx * ARRAY_SIZE + corner_idx) % INPUT_LANES;
                activations_flat[pos] = corners[corner_idx];
            end
            weights_in[WEIGHT_LANES-1] = -8'sd128;
            weights_in[WEIGHT_LANES-2] = -8'sd128;
            activations_flat[INPUT_LANES-1] = -8'sd128;
            activations_flat[INPUT_LANES-2] = -8'sd128;
        end
    endtask

    task automatic run_one_gemm(input int rng_seed, input bit use_corners);
        begin
            fill_random_weights_activations(rng_seed);
            if (use_corners)
                embed_corners();
            apply_reset();
            load_weights();
            run_streamed_gemm(ARRAY_SIZE);
        end
    endtask

    event run_tests_evt;

    initial begin
        @(run_tests_evt);
        run_all_tests();
    end

    task automatic run_all_tests();
        int random_idx;
        begin
            local_mismatches = 0;
            local_gemms = 0;
            run_one_gemm(32'h51E8_0001 + ARRAY_SIZE, 1'b1);
            for (random_idx = 0; random_idx < RANDOM_GEMMS; random_idx++) begin
                run_one_gemm(32'hACED_0000 + (ARRAY_SIZE << 16) + random_idx, 1'b0);
            end
        end
    endtask

    assign mismatch_count = local_mismatches;
    assign gemm_count     = local_gemms;

endmodule

module tb_pe_array_packed;

    localparam int RANDOM_GEMMS = 200;
    localparam int GEMMS_PER_SHAPE = 1 + RANDOM_GEMMS;
    localparam int SHAPE_0 = 8;
    localparam int SHAPE_1 = 16;

    logic clk;
    int mismatch_total;
    int gemm_total;
    int mismatches_8;
    int mismatches_16;
    int gemms_8;
    int gemms_16;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    packed_array_gemm_harness #(
        .ARRAY_SIZE(SHAPE_0)
    ) u_harness_8 (
        .clk(clk),
        .mismatch_count(mismatches_8),
        .gemm_count(gemms_8)
    );

    packed_array_gemm_harness #(
        .ARRAY_SIZE(SHAPE_1)
    ) u_harness_16 (
        .clk(clk),
        .mismatch_count(mismatches_16),
        .gemm_count(gemms_16)
    );

    initial begin
        mismatch_total = 0;
        gemm_total = 0;

        -> u_harness_8.run_tests_evt;
        wait (gemms_8 == GEMMS_PER_SHAPE);
        -> u_harness_16.run_tests_evt;
        wait (gemms_16 == GEMMS_PER_SHAPE);

        mismatch_total = mismatches_8 + mismatches_16;
        gemm_total = gemms_8 + gemms_16;

        if (mismatch_total == 0) begin
            $display("TB_RESULT: PASS shapes=8,16 gemms=%0d", gemm_total);
            $display("PACKED_ARRAY_GEMMS=%0d", gemm_total);
            $display("PACKED_ARRAY_SHAPES=8,16");
        end else begin
            $display("TB_RESULT: FAIL mismatches=%0d gemms=%0d", mismatch_total, gemm_total);
            $fatal(1, "packed array GEMM mismatch");
        end
        $finish;
    end

endmodule
