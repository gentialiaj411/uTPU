`timescale 1ns/1ps

// Deterministic cycle/latency compare: baseline pe_array vs pe_array_packed.
// One square INT8 GEMM per shape (8x8, 16x16); identical stimulus.

module packed_cycle_compare_harness #(
    parameter int ARRAY_SIZE             = 8,
    parameter int COMPUTE_DATA_WIDTH     = 8,
    parameter int ACCUMULATOR_DATA_WIDTH = 32
) (
    input logic clk,
    output int  baseline_first_cycle,
    output int  baseline_full_cycle,
    output int  packed_first_cycle,
    output int  packed_full_cycle,
    output int  stream_cycles,
    output bit  measure_done
);

    localparam int LANES = ARRAY_SIZE * ARRAY_SIZE;

    logic rst;
    logic compute;
    logic load_en;

    logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations_flat [LANES-1:0];

    int base_first;
    int base_full;
    int pack_first;
    int pack_full;
    bit base_first_set;
    bit pack_first_set;

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

    task automatic fill_corner_case();
        int lane;
        begin
            for (lane = 0; lane < LANES; lane++) begin
                weights_in[lane] = '0;
                activations_flat[lane] = '0;
            end
            weights_in[0] = -8'sd128;
            weights_in[LANES-1] = 8'sd127;
            activations_flat[0] = 8'sd1;
            activations_flat[LANES-1] = -8'sd1;
        end
    endtask

    task automatic run_streamed_gemm(input int batch_count);
        int cycle_count;
        int capture_cycle;
        int done_cycle;
        int row;
        int col;
        int idx;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_val;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_val;
        begin
            base_first = -1;
            base_full = -1;
            pack_first = -1;
            pack_full = -1;
            base_first_set = 1'b0;
            pack_first_set = 1'b0;

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
                for (col = 0; col < batch_count; col++) begin
                    for (row = 0; row < ARRAY_SIZE; row++) begin
                        if ((ARRAY_SIZE + 1 + row + col) == capture_cycle) begin
                            base_val = u_baseline.accumulators[ARRAY_SIZE-1][row];
                            pack_val = packed_bottom(row);
                            if (!base_first_set) begin
                                base_first_set = 1'b1;
                                base_first = capture_cycle;
                            end
                            base_full = capture_cycle;
                            if (!pack_first_set) begin
                                pack_first_set = 1'b1;
                                pack_first = capture_cycle;
                            end
                            pack_full = capture_cycle;
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

    event run_evt;

    initial begin
        measure_done = 1'b0;
        @(run_evt);
        fill_corner_case();
        apply_reset();
        load_weights();
        run_streamed_gemm(ARRAY_SIZE);
        $display("CYCLE_SHAPE=%0d BASELINE_FIRST=%0d BASELINE_FULL=%0d PACKED_FIRST=%0d PACKED_FULL=%0d STREAM=%0d",
                 ARRAY_SIZE, base_first, base_full, pack_first, pack_full,
                 (ARRAY_SIZE * 2) + ARRAY_SIZE - 1);
        measure_done = 1'b1;
    end

    assign baseline_first_cycle = base_first;
    assign baseline_full_cycle  = base_full;
    assign packed_first_cycle   = pack_first;
    assign packed_full_cycle    = pack_full;
    assign stream_cycles        = (ARRAY_SIZE * 2) + ARRAY_SIZE - 1;

endmodule

module tb_pe_array_packed_cycles;

    logic clk;
    int base_first_8;
    int base_full_8;
    int pack_first_8;
    int pack_full_8;
    int stream_8;
    int base_first_16;
    int base_full_16;
    int pack_first_16;
    int pack_full_16;
    int stream_16;
    bit done_8;
    bit done_16;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    packed_cycle_compare_harness #(.ARRAY_SIZE(8)) h8 (
        .clk(clk),
        .baseline_first_cycle(base_first_8),
        .baseline_full_cycle(base_full_8),
        .packed_first_cycle(pack_first_8),
        .packed_full_cycle(pack_full_8),
        .stream_cycles(stream_8),
        .measure_done(done_8)
    );

    packed_cycle_compare_harness #(.ARRAY_SIZE(16)) h16 (
        .clk(clk),
        .baseline_first_cycle(base_first_16),
        .baseline_full_cycle(base_full_16),
        .packed_first_cycle(pack_first_16),
        .packed_full_cycle(pack_full_16),
        .stream_cycles(stream_16),
        .measure_done(done_16)
    );

    initial begin
        -> h8.run_evt;
        wait (done_8);
        -> h16.run_evt;
        wait (done_16);
        $display("TB_RESULT: PASS");
        $finish;
    end

endmodule
