`timescale 1ns/1ps

module tb_pe_packed_pair;

    localparam int COMPUTE_DATA_WIDTH     = 8;
    localparam int ACCUMULATOR_DATA_WIDTH = 32;
    localparam int RANDOM_VECTORS         = 1000;
    localparam int RANDOM_SEED            = 16'hACE1;
    localparam int COLUMN_DEPTH           = 8;

    logic clk;
    logic rst;
    logic compute;
    logic load_en;

    logic signed [COMPUTE_DATA_WIDTH-1:0] c_stim;
    logic signed [COMPUTE_DATA_WIDTH-1:0] a_stim;
    logic signed [COMPUTE_DATA_WIDTH-1:0] b_stim;

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_psum_a_in;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_psum_b_in;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_psum_a_out;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] packed_psum_b_out;

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] ref_a_psum_in;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] ref_b_psum_in;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] ref_a_psum_out;
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] ref_b_psum_out;

    int vector_count;
    int mismatch_count;

    pe_packed_pair #(
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) u_packed (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .c_in(c_stim),
        .a_in(a_stim),
        .b_in(b_stim),
        .partial_sum_a_in(packed_psum_a_in),
        .partial_sum_b_in(packed_psum_b_in),
        .c_out(),
        .partial_sum_a_out(packed_psum_a_out),
        .partial_sum_b_out(packed_psum_b_out)
    );

    pe #(
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) u_ref_a (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .data_in(c_stim),
        .weight_in(a_stim),
        .partial_sum_in(ref_a_psum_in),
        .data_out(),
        .partial_sum_out(ref_a_psum_out)
    );

    pe #(
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
    ) u_ref_b (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .data_in(c_stim),
        .weight_in(b_stim),
        .partial_sum_in(ref_b_psum_in),
        .data_out(),
        .partial_sum_out(ref_b_psum_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic reset_dut();
        rst = 1'b1;
        compute = 1'b0;
        load_en = 1'b0;
        c_stim = '0;
        a_stim = '0;
        b_stim = '0;
        packed_psum_a_in = '0;
        packed_psum_b_in = '0;
        ref_a_psum_in = '0;
        ref_b_psum_in = '0;
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;
        @(posedge clk);
    endtask

    task automatic load_operands(
        input logic signed [COMPUTE_DATA_WIDTH-1:0] a_val,
        input logic signed [COMPUTE_DATA_WIDTH-1:0] b_val
    );
        a_stim = a_val;
        b_stim = b_val;
        load_en = 1'b1;
        compute = 1'b0;
        @(posedge clk);
        load_en = 1'b0;
        @(posedge clk);
    endtask

    task automatic mac_cycle(
        input logic signed [COMPUTE_DATA_WIDTH-1:0] c_val,
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] psum_a,
        input logic signed [ACCUMULATOR_DATA_WIDTH-1:0] psum_b
    );
        c_stim = c_val;
        packed_psum_a_in = psum_a;
        packed_psum_b_in = psum_b;
        ref_a_psum_in = psum_a;
        ref_b_psum_in = psum_b;
        compute = 1'b1;
        @(posedge clk);
        compute = 1'b0;
        @(posedge clk);
    endtask

    function automatic void check_outputs(string label);
        if (packed_psum_a_out !== ref_a_psum_out || packed_psum_b_out !== ref_b_psum_out) begin
            mismatch_count++;
            $display("MISMATCH [%s] a=%0d b=%0d c=%0d packed=(%0d,%0d) ref=(%0d,%0d)",
                     label, a_stim, b_stim, c_stim,
                     packed_psum_a_out, packed_psum_b_out,
                     ref_a_psum_out, ref_b_psum_out);
        end
        vector_count++;
    endfunction

    task automatic run_mac_vector(
        input logic signed [COMPUTE_DATA_WIDTH-1:0] a_val,
        input logic signed [COMPUTE_DATA_WIDTH-1:0] b_val,
        input logic signed [COMPUTE_DATA_WIDTH-1:0] c_val,
        input string label
    );
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] zero_psum;
        zero_psum = '0;
        load_operands(a_val, b_val);
        mac_cycle(c_val, zero_psum, zero_psum);
        check_outputs(label);
    endtask

    task automatic run_sign_combos();
        logic signed [COMPUTE_DATA_WIDTH-1:0] mag_a;
        logic signed [COMPUTE_DATA_WIDTH-1:0] mag_b;
        logic signed [COMPUTE_DATA_WIDTH-1:0] mag_c;
        int sa;
        int sb;
        int sc;
        mag_a = 8'sd37;
        mag_b = 8'sd53;
        mag_c = 8'sd19;
        for (sa = 0; sa < 2; sa++) begin
            for (sb = 0; sb < 2; sb++) begin
                for (sc = 0; sc < 2; sc++) begin
                    run_mac_vector(
                        (sa ? -mag_a : mag_a),
                        (sb ? -mag_b : mag_b),
                        (sc ? -mag_c : mag_c),
                        "sign_combo"
                    );
                end
            end
        end
    endtask

    task automatic run_corners();
        logic signed [COMPUTE_DATA_WIDTH-1:0] corners [0:4];
        int i;
        int j;
        int k;
        corners[0] = -8'sd128;
        corners[1] = -8'sd1;
        corners[2] = 8'sd0;
        corners[3] = 8'sd1;
        corners[4] = 8'sd127;
        for (i = 0; i < 5; i++) begin
            for (j = 0; j < 5; j++) begin
                for (k = 0; k < 5; k++) begin
                    run_mac_vector(corners[i], corners[j], corners[k], "corner");
                end
            end
        end
    endtask

    task automatic run_random_sweep();
        int idx;
        int lane;
        int rng_state;
        logic signed [COMPUTE_DATA_WIDTH-1:0] rand_a;
        logic signed [COMPUTE_DATA_WIDTH-1:0] rand_b;
        logic signed [COMPUTE_DATA_WIDTH-1:0] rand_c;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] acc_a;
        logic signed [ACCUMULATOR_DATA_WIDTH-1:0] acc_b;

        rng_state = int'(RANDOM_SEED);
        for (idx = 0; idx < RANDOM_VECTORS; idx++) begin
            rng_state = (rng_state * 32'd1664525 + 32'd1013904223);
            rand_a = rng_state[COMPUTE_DATA_WIDTH-1:0];
            rng_state = (rng_state * 32'd1664525 + 32'd1013904223);
            rand_b = rng_state[COMPUTE_DATA_WIDTH-1:0];
            load_operands(rand_a, rand_b);

            acc_a = '0;
            acc_b = '0;
            for (lane = 0; lane < COLUMN_DEPTH; lane++) begin
                rng_state = (rng_state * 32'd1664525 + 32'd1013904223);
                rand_c = rng_state[COMPUTE_DATA_WIDTH-1:0];
                mac_cycle(rand_c, acc_a, acc_b);
                check_outputs("random_depth8");
                acc_a = packed_psum_a_out;
                acc_b = packed_psum_b_out;
            end
        end
    endtask

    initial begin
        vector_count = 0;
        mismatch_count = 0;

        reset_dut();
        run_sign_combos();
        run_corners();
        run_random_sweep();

        if (mismatch_count == 0) begin
            $display("TB_RESULT: PASS vectors=%0d", vector_count);
            $display("PACKED_PAIR_VECTORS=%0d", vector_count);
        end else begin
            $display("TB_RESULT: FAIL mismatches=%0d vectors=%0d", mismatch_count, vector_count);
            $fatal(1, "packed pair mismatch");
        end
        $finish;
    end

endmodule
