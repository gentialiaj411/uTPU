`timescale 1ns/1ps
`include "build/test_vectors/fused_expected.svh"

module tb_fused_compressed_program;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = 16;
    localparam int TB_BUFFER_SIZE = 512;
    localparam int TB_FIFO_WIDTH = 256;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [7:0] MAGIC_REARM  = 8'hA3;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    int tests = 0;
    int errors = 0;
    int cycle_ctr = 0;

    integer trace_fd;
    int first_failure_cycle = -1;
    reg [15:0] first_failure_instruction = 16'h0000;
    integer first_failure_stage = 0; // 0 none, 1 upload, 2 start, 3 bstore, 4 load, 5 run, 6 finalize, 7 fetch

    byte case1_expected [0:`CASE1_FETCH_N-1];
    byte case2_expected [0:`CASE2_FETCH_N-1];
    byte case3_expected [0:`CASE3_FETCH_N-1];
    byte case1_actual [0:63];
    byte case2_actual [0:63];
    byte case3_actual [0:63];
    int case1_actual_n;
    int case2_actual_n;
    int case3_actual_n;
    bit case1_passed;
    bit case2_passed;
    bit case3_passed;

    reg [15:0] case_mem [0:1023];

    task automatic CHECK(input string name, input bit cond);
        tests++;
        if (!cond) begin
            errors++;
            $display("[FAIL] %s", name);
        end else begin
            $display("[ OK ] %s", name);
        end
    endtask

    task automatic mark_failure(input integer stage_id);
        if (first_failure_stage == 0) begin
            first_failure_stage = stage_id;
            first_failure_cycle = cycle_ctr;
            first_failure_instruction = dut.instruction;
        end
    endtask

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    task push_rx_byte(input logic [7:0] b);
        @(posedge clk);
        dut.fifo_in.mem[dut.fifo_in.w_ptr[TB_FIFO_PTR_W-1:0]] = b;
        dut.fifo_in.w_ptr = dut.fifo_in.w_ptr + 1'b1;
        @(posedge clk);
    endtask

    task automatic stream_program(input int words);
        int i;
        logic [15:0] w;
        // Re-arm if needed, then upload + length.
        push_rx_byte(MAGIC_REARM);
        push_rx_byte(MAGIC_UPLOAD);
        push_rx_byte(words[7:0]);
        push_rx_byte(words[15:8]);
        for (i = 0; i < words; i++) begin
            w = case_mem[i];
            push_rx_byte(w[7:0]);
            push_rx_byte(w[15:8]);
        end
    endtask

    task automatic wait_halt(input int max_cycles, output bit halted);
        int i;
        bit done;
        halted = 1'b0;
        done = 1'b0;
        for (i = 0; i < max_cycles; i++) begin
            @(posedge clk);
            if (!done && dut.current_state == dut.HALT_STATE) begin
                halted = 1'b1;
                done = 1'b1;
            end
        end
    endtask

    task automatic wait_wait_start(input int max_cycles, output bit ok);
        int i;
        ok = 1'b0;
        i = 0;
        while (i < max_cycles && !ok) begin
            @(posedge clk);
            if (dut.current_state == dut.WAIT_START_STATE) begin
                ok = 1'b1;
            end
            i = i + 1;
        end
    endtask

    task automatic run_case(
        input string case_name,
        input string mem_path,
        input int prog_words,
        input int exp_n,
        input int case_id
    );
        int i;
        bit halted;
        byte exp_byte;
        byte act_byte;
        int local_count;
        int mismatch_count;
        bit saw_wait_start;

        $display("---- %s ----", case_name);
        for (i = 0; i < 1024; i++) case_mem[i] = 16'h0000;
        $readmemh(mem_path, case_mem);

        stream_program(prog_words);
        wait_wait_start(200000, saw_wait_start);
        CHECK({case_name, " reached WAIT_START"}, saw_wait_start);
        if (!saw_wait_start) mark_failure(1);
        push_rx_byte(MAGIC_START);

        local_count = 0;
        while (local_count < exp_n && cycle_ctr < 500000) begin
            @(posedge clk);
            // Capture actual transmitted UART FIFO bytes, not pending shadow registers.
            if (dut.tx_we) begin
                // Ignore one-time self-test AA and any non-fetch noise.
                if (dut.tx_wdata !== 8'hAA) begin
                    if (case_id == 1) case1_actual[local_count] = dut.tx_wdata;
                    else if (case_id == 2) case2_actual[local_count] = dut.tx_wdata;
                    else case3_actual[local_count] = dut.tx_wdata;
                    local_count = local_count + 1;
                end
            end
        end
        if (case_id == 1) case1_actual_n = local_count;
        else if (case_id == 2) case2_actual_n = local_count;
        else case3_actual_n = local_count;

        wait_halt(20000, halted);
        CHECK({case_name, " HALT reached"}, halted);
        if (!halted) mark_failure(2);
        CHECK({case_name, " fetch byte count"}, local_count == exp_n);
        if (local_count != exp_n) mark_failure(7);

        mismatch_count = 0;
        for (i = 0; i < exp_n; i++) begin
            if (case_id == 1) begin
                exp_byte = case1_expected[i];
                act_byte = case1_actual[i];
            end else if (case_id == 2) begin
                exp_byte = case2_expected[i];
                act_byte = case2_actual[i];
            end else begin
                exp_byte = case3_expected[i];
                act_byte = case3_actual[i];
            end
            CHECK($sformatf("%s byte %0d", case_name, i), act_byte === exp_byte);
            if (act_byte !== exp_byte) begin
                mismatch_count = mismatch_count + 1;
                mark_failure(7);
            end
        end

        $display("%s byte mismatches: %0d / %0d", case_name, mismatch_count, exp_n);
        if (case_id == 1) case1_passed = halted && (local_count == exp_n) && (mismatch_count == 0);
        else if (case_id == 2) case2_passed = halted && (local_count == exp_n) && (mismatch_count == 0);
        else case3_passed = halted && (local_count == exp_n) && (mismatch_count == 0);
    endtask

    always @(posedge clk) begin
        cycle_ctr <= cycle_ctr + 1;

        // Instruction-level trace.
        if (dut.current_state == dut.DECODE_STATE ||
            dut.current_state == dut.BSTORE_FETCH_COUNT_STATE ||
            dut.current_state == dut.BSTORE_FETCH_DATA_STATE ||
            dut.current_state == dut.BSTORE_WRITE_STATE ||
            dut.current_state == dut.LOAD_STATE ||
            dut.current_state == dut.COMPUTE_STATE ||
            dut.current_state == dut.FETCH_BUFFER_STATE) begin
            $fdisplay(trace_fd,
                "cycle=%0d state=%0d pc=%0d instr=%04h opcode=%0d cdone=%0d bcount=%0d bidx=%0d baddr=%0d bdata=%04h load_addr=%0d run[c=%0d q=%0d r=%0d clr=%0d ld=%0d] m0=%0d ci0=[%0d,%0d,%0d,%0d] ci_col0=[%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d] ci240=[%0d,%0d,%0d,%0d] co=[%0d,%0d,%0d,%0d] pcc=%0d pd0=%0d pr0=%0d pra0=%0d pe_w0=%0d pe_w15=%0d pe_prod0=%0d pe_ps0=%0d row_ps=[%0d,%0d,%0d,%0d] row15_c=[%0d,%0d,%0d,%0d] run_cap0=%0d ctb0=%0d acc0=%0d acc1=%0d fetch_addr=%0d tx_pending=%0d tx_data=%02h ub_re=%0d ub_done=%0d ub_rc=%0d ub_base=%0d ub_row=%0d ub_d0=%04h",
                cycle_ctr, dut.current_state, dut.pc, dut.instruction, dut.opcode,
                dut.compute_done,
                dut.bstore_count, dut.bstore_index, dut.address, dut.bstore_data_word,
                dut.address, dut.compute_en, dut.quantizer_en, dut.relu_en, dut.acc_clear_en, dut.compute_load_en,
                dut.mem_to_compute[0],
                dut.compute_stream_in[0], dut.compute_stream_in[1], dut.compute_stream_in[2], dut.compute_stream_in[3],
                dut.compute_stream_in[0], dut.compute_stream_in[16], dut.compute_stream_in[32], dut.compute_stream_in[48],
                dut.compute_stream_in[64], dut.compute_stream_in[80], dut.compute_stream_in[96], dut.compute_stream_in[112],
                dut.compute_stream_in[128], dut.compute_stream_in[144], dut.compute_stream_in[160], dut.compute_stream_in[176],
                dut.compute_stream_in[192], dut.compute_stream_in[208], dut.compute_stream_in[224], dut.compute_stream_in[240],
                dut.compute_stream_in[240], dut.compute_stream_in[241], dut.compute_stream_in[242], dut.compute_stream_in[243],
                dut.compute_stream_out[0], dut.compute_stream_out[1], dut.compute_stream_out[2], dut.compute_stream_out[3],
                dut.u_pe_array.cycle_count, dut.u_pe_array.datas_in[0], dut.u_pe_array.results[0], dut.u_pe_array.results_arr[0],
                dut.u_pe_array.u_pe_array.gen_rows[0].gen_cols[0].gen_top_row.u_pe.weight,
                dut.u_pe_array.u_pe_array.gen_rows[15].gen_cols[0].gen_non_top_row.u_pe.weight,
                dut.u_pe_array.u_pe_array.gen_rows[0].gen_cols[0].gen_top_row.u_pe.prod,
                dut.u_pe_array.u_pe_array.gen_rows[0].gen_cols[0].gen_top_row.u_pe.partial_sum_out,
                dut.u_pe_array.u_pe_array.accumulators[0][0],
                dut.u_pe_array.u_pe_array.accumulators[1][0],
                dut.u_pe_array.u_pe_array.accumulators[2][0],
                dut.u_pe_array.u_pe_array.accumulators[15][0],
                dut.u_pe_array.u_pe_array.accumulators[15][0],
                dut.u_pe_array.u_pe_array.accumulators[15][1],
                dut.u_pe_array.u_pe_array.accumulators[15][2],
                dut.u_pe_array.u_pe_array.accumulators[15][3],
                dut.run_capture_sums[0],
                dut.compute_to_buffer[0],
                dut.acc_partial_sums[0], dut.acc_partial_sums[1], dut.address, dut.tx_pending, dut.tx_pending_data,
                dut.u_unified_buffer.re, dut.u_unified_buffer.done, dut.u_unified_buffer.read_compute_d,
                dut.u_unified_buffer.base_bank, dut.u_unified_buffer.base_row, dut.u_unified_buffer.bank_dout[0]
            );
        end
    end

    initial begin
`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_fused_compressed_program.vcd");
        $dumpvars(0, tb_fused_compressed_program);
`endif
        trace_fd = $fopen("build/reports/rtl_fused_trace.log", "w");
        if (trace_fd == 0) begin
            $display("TRACE_OPEN_FAIL");
            $finish;
        end

        case1_expected[0] = `CASE1_EXP_BYTE_0;
        case1_expected[1] = `CASE1_EXP_BYTE_1;
        case1_expected[2] = `CASE1_EXP_BYTE_2;
        case1_expected[3] = `CASE1_EXP_BYTE_3;
        case1_expected[4] = `CASE1_EXP_BYTE_4;
        case1_expected[5] = `CASE1_EXP_BYTE_5;
        case1_expected[6] = `CASE1_EXP_BYTE_6;
        case1_expected[7] = `CASE1_EXP_BYTE_7;
        case2_expected[0] = `CASE2_EXP_BYTE_0;
        case2_expected[1] = `CASE2_EXP_BYTE_1;
        case2_expected[2] = `CASE2_EXP_BYTE_2;
        case2_expected[3] = `CASE2_EXP_BYTE_3;
        case2_expected[4] = `CASE2_EXP_BYTE_4;
        case2_expected[5] = `CASE2_EXP_BYTE_5;
        case2_expected[6] = `CASE2_EXP_BYTE_6;
        case2_expected[7] = `CASE2_EXP_BYTE_7;
        case3_expected[0] = `CASE3_EXP_BYTE_0;
        case3_expected[1] = `CASE3_EXP_BYTE_1;
        case3_expected[2] = `CASE3_EXP_BYTE_2;
        case3_expected[3] = `CASE3_EXP_BYTE_3;
        case3_expected[4] = `CASE3_EXP_BYTE_4;
        case3_expected[5] = `CASE3_EXP_BYTE_5;
        case3_expected[6] = `CASE3_EXP_BYTE_6;
        case3_expected[7] = `CASE3_EXP_BYTE_7;
        case1_actual_n = 0;
        case2_actual_n = 0;
        case3_actual_n = 0;
        case1_passed = 1'b0;
        case2_passed = 1'b0;
        case3_passed = 1'b0;

        $display("=================================================");
        $display("fused compressed RTL decode/control testbench");
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        run_case(`CASE1_NAME, `CASE1_MEM, `CASE1_WORDS, `CASE1_FETCH_N, 1);
        run_case(`CASE2_NAME, `CASE2_MEM, `CASE2_WORDS, `CASE2_FETCH_N, 2);
        run_case(`CASE3_NAME, `CASE3_MEM, `CASE3_WORDS, `CASE3_FETCH_N, 3);

        $display("CASE1_ACTUAL_BYTES=%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x", case1_actual[0], case1_actual[1], case1_actual[2], case1_actual[3], case1_actual[4], case1_actual[5], case1_actual[6], case1_actual[7]);
        $display("CASE2_ACTUAL_BYTES=%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x", case2_actual[0], case2_actual[1], case2_actual[2], case2_actual[3], case2_actual[4], case2_actual[5], case2_actual[6], case2_actual[7]);
        $display("CASE3_ACTUAL_BYTES=%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x", case3_actual[0], case3_actual[1], case3_actual[2], case3_actual[3], case3_actual[4], case3_actual[5], case3_actual[6], case3_actual[7]);
        $display("CASE1_PASS=%0d", case1_passed);
        $display("CASE2_PASS=%0d", case2_passed);
        $display("CASE3_PASS=%0d", case3_passed);
        $display("FIRST_FAILURE_STAGE=%0d", first_failure_stage);
        $display("FIRST_FAILURE_CYCLE=%0d", first_failure_cycle);
        $display("FIRST_FAILURE_INSTRUCTION=%04h", first_failure_instruction);
        $display("TOTAL_CYCLES=%0d", cycle_ctr);
        $display("TRACE_LOG_PATH=build/reports/rtl_fused_trace.log");

        $display("=================================================");
        $display("DONE tests=%0d errors=%0d", tests, errors);
        $display("=================================================");
        if (errors != 0) begin
            $fclose(trace_fd);
            $fatal(1, "tb_fused_compressed_program FAILED");
        end
        $fclose(trace_fd);
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
