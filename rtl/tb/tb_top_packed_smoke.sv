`timescale 1ns/1ps

module top_packed_smoke_harness #(
    parameter int ARRAY_SIZE             = 8,
    parameter int COMPUTE_DATA_WIDTH     = 8,
    parameter int ACCUMULATOR_DATA_WIDTH = 32,
    parameter int MAX_BATCH_COUNT        = 64
) (
    input logic clk,
    output int mismatch_count,
    output bit tests_done
);

    localparam int LANES = ARRAY_SIZE * ARRAY_SIZE;
    localparam int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH_COUNT + 1);

    logic rst;
    logic compute;
    logic load_en;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count;
    logic base_done;
    logic pack_done;
    logic requant_enable;

    logic signed [COMPUTE_DATA_WIDTH-1:0] weights_in [LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] datas_arr [ARRAY_SIZE*MAX_BATCH_COUNT-1:0];
    logic [(LANES*16)-1:0] requant_multiplier_flat;
    logic [(LANES*16)-1:0] requant_right_shift_flat;

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_stream [ARRAY_SIZE*MAX_BATCH_COUNT-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] base_accum [LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] base_quant [LANES-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] pack_accum [LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] pack_quant [LANES-1:0];
    logic [(LANES*ACCUMULATOR_DATA_WIDTH)-1:0] base_accum_flat;
    logic [(LANES*COMPUTE_DATA_WIDTH)-1:0] base_quant_flat;

    int mismatches;

    pe_controller #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH)
    ) u_baseline (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .batch_count(batch_count),
        .done(base_done),
        .datas_arr(datas_arr),
        .weights_in(weights_in),
        .results_arr(base_stream)
    );

    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi++) begin : base_pack
            assign base_accum[gi] = base_stream[gi];
            assign base_accum_flat[(gi*ACCUMULATOR_DATA_WIDTH) +: ACCUMULATOR_DATA_WIDTH] = base_accum[gi];
            assign base_quant[gi] = base_quant_flat[(gi*COMPUTE_DATA_WIDTH) +: COMPUTE_DATA_WIDTH];
        end
    endgenerate

    quantizer_array #(
        .QUANTIZER_SIZE(LANES),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
    ) u_base_quant (
        .ins_flat(base_accum_flat),
        .requant_enable(requant_enable),
        .requant_multiplier_flat(requant_multiplier_flat),
        .requant_right_shift_flat(requant_right_shift_flat),
        .results_flat(base_quant_flat)
    );

    top_packed #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT)
    ) u_packed_top (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .batch_count(batch_count),
        .compute_done(pack_done),
        .weights_in(weights_in),
        .datas_arr(datas_arr),
        .requant_enable(requant_enable),
        .requant_multiplier_flat(requant_multiplier_flat),
        .requant_right_shift_flat(requant_right_shift_flat),
        .accum_results(pack_accum),
        .quant_results(pack_quant)
    );

    task automatic apply_reset();
        rst = 1'b1;
        compute = 1'b0;
        load_en = 1'b0;
        requant_enable = 1'b0;
        batch_count = ARRAY_SIZE[BATCH_COUNT_WIDTH-1:0];
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

    task automatic fill_case(input int seed);
        int state;
        int lane;
        begin
            state = seed;
            for (lane = 0; lane < LANES; lane++) begin
                state = (state * 32'd1664525 + 32'd1013904223);
                weights_in[lane] = state[COMPUTE_DATA_WIDTH-1:0];
            end
            for (lane = 0; lane < LANES; lane++) begin
                state = (state * 32'd1664525 + 32'd1013904223);
                datas_arr[lane] = state[COMPUTE_DATA_WIDTH-1:0];
            end
            requant_multiplier_flat = {(LANES*16){1'b0}};
            requant_right_shift_flat = {(LANES*16){1'b0}};
            for (lane = 0; lane < LANES; lane++) begin
                requant_multiplier_flat[(lane*16)+:16] = 16'd1;
            end
        end
    endtask

    task automatic run_compute();
        compute = 1'b1;
        while (!base_done || !pack_done)
            @(posedge clk);
        compute = 1'b0;
        requant_enable = 1'b1;
        @(posedge clk);
        requant_enable = 1'b0;
        @(posedge clk);
    endtask

    task automatic compare_outputs();
        int lane;
        begin
            for (lane = 0; lane < LANES; lane++) begin
                if (base_accum[lane] !== pack_accum[lane]) begin
                    mismatches++;
                    $display("ACC_MISMATCH shape=%0d lane=%0d base=%0d pack=%0d",
                             ARRAY_SIZE, lane, base_accum[lane], pack_accum[lane]);
                end
                if (base_quant[lane] !== pack_quant[lane]) begin
                    mismatches++;
                    $display("QUANT_MISMATCH shape=%0d lane=%0d base=%0d pack=%0d",
                             ARRAY_SIZE, lane, base_quant[lane], pack_quant[lane]);
                end
            end
        end
    endtask

    event run_evt;

    initial begin
        tests_done = 1'b0;
        @(run_evt);
        mismatches = 0;
        fill_case(32'h7A50_0000 + ARRAY_SIZE);
        apply_reset();
        load_weights();
        run_compute();
        compare_outputs();
        $display("TOP_PACKED_SMOKE shape=%0d mismatches=%0d", ARRAY_SIZE, mismatches);
        tests_done = 1'b1;
    end

    assign mismatch_count = mismatches;

endmodule

module tb_top_packed_smoke;

    logic clk;
    int mismatches_8;
    int mismatches_16;
    bit done_8;
    bit done_16;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    top_packed_smoke_harness #(.ARRAY_SIZE(8)) h8 (
        .clk(clk),
        .mismatch_count(mismatches_8),
        .tests_done(done_8)
    );

    top_packed_smoke_harness #(.ARRAY_SIZE(16)) h16 (
        .clk(clk),
        .mismatch_count(mismatches_16),
        .tests_done(done_16)
    );

    initial begin
        -> h8.run_evt;
        wait (done_8);
        -> h16.run_evt;
        wait (done_16);
        if (mismatches_8 == 0 && mismatches_16 == 0) begin
            $display("TB_RESULT: PASS wrapper=full_datapath shapes=8,16");
            $display("TOP_PACKED_WRAPPER=full_datapath_requant");
        end else begin
            $display("TB_RESULT: FAIL mismatches_8=%0d mismatches_16=%0d", mismatches_8, mismatches_16);
            $fatal(1, "top_packed smoke mismatch");
        end
        $finish;
    end

endmodule
