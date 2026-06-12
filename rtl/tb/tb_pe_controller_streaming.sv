`timescale 1ns/1ps

module tb_pe_controller_streaming;
    localparam int ARRAY_SIZE = 16;
    localparam int MAX_BATCH_COUNT = 64;
    localparam int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH_COUNT + 1);

    logic clk = 0;
    logic rst = 1;
    logic compute = 0;
    logic load_en = 0;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count = '0;
    logic done;
    logic signed [3:0] datas_arr [0:ARRAY_SIZE*MAX_BATCH_COUNT-1];
    logic signed [3:0] weights_in [0:ARRAY_SIZE*ARRAY_SIZE-1];
    logic signed [15:0] results_arr [0:ARRAY_SIZE*MAX_BATCH_COUNT-1];

    int tests = 0;
    int errors = 0;

    always #5 clk = ~clk;

    pe_controller #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .COMPUTE_DATA_WIDTH(4),
        .ACCUMULATOR_DATA_WIDTH(16),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .batch_count(batch_count),
        .done(done),
        .datas_arr(datas_arr),
        .weights_in(weights_in),
        .results_arr(results_arr)
    );

    task automatic CHECK(input string name, input bit cond);
        tests++;
        if (!cond) begin
            errors++;
            $display("[FAIL] %s", name);
        end else begin
            $display("[ OK ] %s", name);
        end
    endtask

    task automatic init_weights;
        int row;
        int col;
        begin
            for (row = 0; row < ARRAY_SIZE; row++) begin
                for (col = 0; col < ARRAY_SIZE; col++) begin
                    weights_in[row*ARRAY_SIZE + col] = ((row * 3 + col * 5) % 7) - 3;
                end
            end
        end
    endtask

    task automatic init_inputs(input int active_batch);
        int row;
        int col;
        begin
            for (col = 0; col < MAX_BATCH_COUNT; col++) begin
                for (row = 0; row < ARRAY_SIZE; row++) begin
                    if (col < active_batch)
                        datas_arr[col*ARRAY_SIZE + row] = ((col * 2 + row * 3) % 8) - 4;
                    else
                        datas_arr[col*ARRAY_SIZE + row] = '0;
                end
            end
        end
    endtask

    task automatic run_case(input int active_batch);
        int row;
        int col;
        int k;
        int cycles;
        int expected;
        int expected_cycles;
        bit hit_done;
        begin
            init_inputs(active_batch);
            batch_count = active_batch[BATCH_COUNT_WIDTH-1:0];
            compute = 1'b1;
            cycles = 0;
            hit_done = 1'b0;
            while (!hit_done && cycles < 512) begin
                @(posedge clk);
                cycles++;
                if (done)
                    hit_done = 1'b1;
            end
            compute = 1'b0;
            @(posedge clk);

            expected_cycles = (2 * ARRAY_SIZE) + active_batch;
            $display("STREAM_CASE B=%0d cycles=%0d expected_cycles=%0d sample0=%0d sample1=%0d",
                active_batch, cycles, expected_cycles, results_arr[0], results_arr[1]);
            CHECK($sformatf("done asserted for B=%0d", active_batch), hit_done);
            CHECK($sformatf("cycle law for B=%0d", active_batch), cycles == expected_cycles);
            for (col = 0; col < active_batch; col++) begin
                for (row = 0; row < ARRAY_SIZE; row++) begin
                    expected = 0;
                    for (k = 0; k < ARRAY_SIZE; k++) begin
                        expected += weights_in[row*ARRAY_SIZE + k] * datas_arr[col*ARRAY_SIZE + k];
                    end
                    CHECK(
                        $sformatf("result row=%0d col=%0d batch=%0d", row, col, active_batch),
                        results_arr[col*ARRAY_SIZE + row] === expected
                    );
                end
            end
        end
    endtask

    initial begin
        int batches [0:4];
        batches[0] = 1;
        batches[1] = 4;
        batches[2] = 16;
        batches[3] = 32;
        batches[4] = 64;

        init_weights();
        for (int i = 0; i < ARRAY_SIZE*MAX_BATCH_COUNT; i++)
            datas_arr[i] = '0;

        repeat (4) @(posedge clk);
        rst = 0;
        load_en = 1'b1;
        @(posedge clk);
        load_en = 1'b0;
        @(posedge clk);

        for (int bi = 0; bi < 5; bi++) begin
            run_case(batches[bi]);
        end

        if (errors != 0) begin
            $display("TB_RESULT: FAIL");
            $fatal(1, "tb_pe_controller_streaming FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
