`timescale 1ns/1ps
`include "build/test_vectors/mnist_utpu_expected.svh"

module tb_mnist_utpu_program;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = 16;
    localparam int TB_BUFFER_SIZE = 1024;
    localparam int TB_FIFO_WIDTH = 256;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);
`ifdef MNIST_CASE_LIMIT
    localparam int TB_CASE_LIMIT = `MNIST_CASE_LIMIT;
`else
    localparam int TB_CASE_LIMIT = 3;
`endif
    localparam int TB_FIFO_MARGIN = 32;
    localparam int TB_PER_CASE_BUDGET = 0;
    localparam int TB_TOTAL_CYCLE_WATCHDOG = 2000000;
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [7:0] MAGIC_REARM  = 8'hA3;
    localparam logic [7:0] MAGIC_READ_PERF = 8'hA4;
    localparam int TB_PERF_BYTES = 104;
    // Match tb_batched_gemm when MNIST_ATTR is set so result_fetch shares
    // are comparable on-chip-core-cycle numbers (not shipping 115200 baud).
`ifdef MNIST_ATTR
    localparam int TB_UART_BAUD = 100000000;
    localparam int TB_PERF_WAIT_MAX = 2000000;
`else
    localparam int TB_UART_BAUD = 115200;
    localparam int TB_PERF_WAIT_MAX = 2000000;
`endif

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .PROG_DEPTH(`TB_PROG_DEPTH),
        .EXT_ADDR_EN(1),
        .UART_INPUT_CLK(100000000),
        .UART_BAUD(TB_UART_BAUD)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    int tests = 0;
    int errors = 0;
    int cycle_ctr = 0;

    integer trace_fd;
    int first_failure_cycle = -1;
    reg [15:0] first_failure_instruction = 16'h0000;
    integer first_failure_stage = 0;

    byte case1_expected [0:`CASE1_FETCH_N-1];
    byte case2_expected [0:`CASE2_FETCH_N-1];
    byte case3_expected [0:`CASE3_FETCH_N-1];
    int case1_actual_n;
    int case2_actual_n;
    int case3_actual_n;
    bit case1_passed;
    bit case2_passed;
    bit case3_passed;

    reg [15:0] case_mem [0:4095];
    byte perf_bytes [0:103];

    function automatic byte expected_byte(input string case_name, input int idx);
        if (case_name == `CASE1_NAME) expected_byte = case1_expected[idx];
        else if (case_name == `CASE2_NAME) expected_byte = case2_expected[idx];
        else expected_byte = case3_expected[idx];
    endfunction

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
        int fifo_occupancy;
        fifo_occupancy = (dut.fifo_in.w_ptr - dut.fifo_in.r_ptr) & ((1 << (TB_FIFO_PTR_W + 1)) - 1);
        while (dut.fifo_in.full) begin
            @(posedge clk);
            fifo_occupancy = (dut.fifo_in.w_ptr - dut.fifo_in.r_ptr) & ((1 << (TB_FIFO_PTR_W + 1)) - 1);
        end
        @(posedge clk);
        dut.fifo_in.mem[dut.fifo_in.w_ptr[TB_FIFO_PTR_W-1:0]] = b;
        dut.fifo_in.w_ptr = dut.fifo_in.w_ptr + 1'b1;
        $display("[UPLOAD] full=%0d occ=%0d upload_count=%0d byte=%02h", dut.fifo_in.full, fifo_occupancy, dut.upload_count, b);
        @(posedge clk);
    endtask

    task automatic stream_program(input int words);
        int i;
        logic [15:0] w;
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

    task automatic collect_perf_bytes(output int got);
        int i;
        got = 0;
        i = 0;
        while (i < TB_PERF_WAIT_MAX && got < TB_PERF_BYTES) begin
            @(posedge clk);
            if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                perf_bytes[got] = dut.tx_wdata;
                got = got + 1;
            end
            i = i + 1;
        end
    endtask

    function automatic logic [63:0] unpack_u64(input int base);
        logic [63:0] v;
        int j;
        v = '0;
        for (j = 0; j < 8; j++)
            v = {v[55:0], perf_bytes[base + j]};
        return v;
    endfunction

    task automatic dump_attr(input string case_name);
        int got_perf;
        logic [63:0] program_cycle_ctr;
        logic [63:0] attr_fetch_decode;
        logic [63:0] attr_store;
        logic [63:0] attr_load;
        logic [63:0] attr_compute;
        logic [63:0] attr_writeback;
        logic [63:0] attr_bstore;
        logic [63:0] attr_ext_addr;
        logic [63:0] attr_requant;
        logic [63:0] attr_result_fetch;
        push_rx_byte(MAGIC_READ_PERF);
        collect_perf_bytes(got_perf);
        program_cycle_ctr = '0;
        attr_fetch_decode = '0;
        attr_store = '0;
        attr_load = '0;
        attr_compute = '0;
        attr_writeback = '0;
        attr_bstore = '0;
        attr_ext_addr = '0;
        attr_requant = '0;
        attr_result_fetch = '0;
        if (got_perf == TB_PERF_BYTES) begin
            program_cycle_ctr = unpack_u64(24);
            attr_fetch_decode = unpack_u64(32);
            attr_store = unpack_u64(40);
            attr_load = unpack_u64(48);
            attr_compute = unpack_u64(56);
            attr_writeback = unpack_u64(64);
            attr_bstore = unpack_u64(72);
            attr_ext_addr = unpack_u64(80);
            attr_requant = unpack_u64(88);
            attr_result_fetch = unpack_u64(96);
        end else begin
            program_cycle_ctr = dut.perf_program_cycle_counter;
            attr_fetch_decode = dut.perf_attr_fetch_decode;
            attr_store = dut.perf_attr_store;
            attr_load = dut.perf_attr_load;
            attr_compute = dut.perf_attr_compute;
            attr_writeback = dut.perf_attr_writeback;
            attr_bstore = dut.perf_attr_bstore;
            attr_ext_addr = dut.perf_attr_ext_addr;
            attr_requant = dut.perf_attr_requant;
            attr_result_fetch = dut.perf_attr_result_fetch;
        end
        $display("%s_TOTAL_PROGRAM_CYCLES=%0d", case_name, program_cycle_ctr);
        $display("%s_ATTR_FETCH_DECODE=%0d", case_name, attr_fetch_decode);
        $display("%s_ATTR_STORE=%0d", case_name, attr_store);
        $display("%s_ATTR_LOAD=%0d", case_name, attr_load);
        $display("%s_ATTR_COMPUTE=%0d", case_name, attr_compute);
        $display("%s_ATTR_WRITEBACK=%0d", case_name, attr_writeback);
        $display("%s_ATTR_BSTORE=%0d", case_name, attr_bstore);
        $display("%s_ATTR_EXT_ADDR=%0d", case_name, attr_ext_addr);
        $display("%s_ATTR_REQUANT=%0d", case_name, attr_requant);
        $display("%s_ATTR_RESULT_FETCH=%0d", case_name, attr_result_fetch);
        $display("TOTAL_PROGRAM_CYCLES=%0d", program_cycle_ctr);
        $display("ATTR_FETCH_DECODE=%0d", attr_fetch_decode);
        $display("ATTR_STORE=%0d", attr_store);
        $display("ATTR_LOAD=%0d", attr_load);
        $display("ATTR_COMPUTE=%0d", attr_compute);
        $display("ATTR_WRITEBACK=%0d", attr_writeback);
        $display("ATTR_BSTORE=%0d", attr_bstore);
        $display("ATTR_EXT_ADDR=%0d", attr_ext_addr);
        $display("ATTR_REQUANT=%0d", attr_requant);
        $display("ATTR_RESULT_FETCH=%0d", attr_result_fetch);
        $display("UART_BAUD_TB=%0d", TB_UART_BAUD);
    endtask

    task automatic wait_wait_start(input int max_cycles, input int start_cycle, output bit ok, output int wait_start_cycle);
        int i;
        ok = 1'b0;
        wait_start_cycle = -1;
        i = 0;
        while (i < max_cycles && !ok) begin
            @(posedge clk);
            if (dut.current_state == dut.WAIT_START_STATE) begin
                ok = 1'b1;
                wait_start_cycle = cycle_ctr - start_cycle;
            end
            i = i + 1;
        end
    endtask

    task automatic run_case(
        input string case_name,
        input string mem_path,
        input int prog_words,
        input int exp_n,
        output int actual_n,
        output bit passed
    );
        int i;
        bit halted;
        byte exp_byte;
        byte act_byte;
        int mismatch_count;
        bit saw_wait_start;
        int start_cycle;
        int wait_start_cycle;
        byte actual_bytes [0:63];

        $display("---- %s ----", case_name);
        for (i = 0; i < 4096; i++) case_mem[i] = 16'h0000;
        $readmemh(mem_path, case_mem);

        stream_program(prog_words);
        start_cycle = cycle_ctr;
        wait_wait_start(50000, start_cycle, saw_wait_start, wait_start_cycle);
        CHECK({case_name, " reached WAIT_START"}, saw_wait_start);
        if (!saw_wait_start) mark_failure(1);
        $display("%s_WAIT_START_CYCLES=%0d", case_name, wait_start_cycle);
        push_rx_byte(MAGIC_START);

        actual_n = 0;
        while (actual_n < exp_n) begin
            @(posedge clk);
            if (dut.tx_we) begin
                if (dut.tx_wdata !== 8'hAA) begin
                    actual_bytes[actual_n] = dut.tx_wdata;
                    actual_n = actual_n + 1;
                end
            end
        end

        wait_halt(50000, halted);
        CHECK({case_name, " HALT reached"}, halted);
        if (!halted) mark_failure(2);
        CHECK({case_name, " fetch byte count"}, actual_n == exp_n);
        if (actual_n != exp_n) mark_failure(7);

        mismatch_count = 0;
        for (i = 0; i < exp_n; i++) begin
            exp_byte = expected_byte(case_name, i);
            act_byte = actual_bytes[i];
            CHECK($sformatf("%s byte %0d", case_name, i), act_byte === exp_byte);
            if (act_byte !== exp_byte) begin
                mismatch_count = mismatch_count + 1;
                mark_failure(7);
            end
        end

        $display("%s byte mismatches: %0d / %0d", case_name, mismatch_count, exp_n);
        $write("%s_ACTUAL_BYTES=", case_name);
        for (i = 0; i < exp_n; i++) begin
            if (i != 0) $write(",");
            $write("%02x", actual_bytes[i]);
        end
        $display("");
        $display("%s_CYCLES=%0d", case_name, cycle_ctr - start_cycle);
        passed = halted && (actual_n == exp_n) && (mismatch_count == 0);
`ifdef MNIST_ATTR
        dump_attr(case_name);
`endif
    endtask

    always @(posedge clk) begin
        cycle_ctr <= cycle_ctr + 1;
`ifdef MNIST_TRACE
        if (dut.current_state == dut.DECODE_STATE ||
            dut.current_state == dut.BSTORE_FETCH_COUNT_STATE ||
            dut.current_state == dut.BSTORE_FETCH_DATA_STATE ||
            dut.current_state == dut.BSTORE_WRITE_STATE ||
            dut.current_state == dut.LOAD_STATE ||
            dut.current_state == dut.COMPUTE_STATE ||
            dut.current_state == dut.FETCH_BUFFER_STATE) begin
            $fdisplay(trace_fd,
                "cycle=%0d state=%0d pc=%0d instr=%04h opcode=%0d cdone=%0d tx_we=%0d tx_wdata=%02h bcount=%0d bidx=%0d baddr=%0d bdata=%04h load_addr=%0d run[c=%0d q=%0d r=%0d clr=%0d] m0=%0d ci0=[%0d,%0d,%0d,%0d] co=[%0d,%0d,%0d,%0d] ub_re=%0d ub_done=%0d ub_rc=%0d ub_base=%0d ub_row=%0d",
                cycle_ctr, dut.current_state, dut.pc, dut.instruction, dut.opcode,
                dut.compute_done,
                dut.tx_we, dut.tx_wdata,
                dut.bstore_count, dut.bstore_index, dut.address, dut.bstore_data_word,
                dut.address, dut.compute_en, dut.quantizer_en, dut.relu_en, dut.acc_clear_en,
                dut.mem_to_compute[0],
                dut.compute_in[0], dut.compute_in[1], dut.compute_in[2], dut.compute_in[3],
                dut.compute_out[0], dut.compute_out[1], dut.compute_out[2], dut.compute_out[3],
                dut.u_unified_buffer.re, dut.u_unified_buffer.done, dut.u_unified_buffer.read_compute_d,
                dut.u_unified_buffer.base_bank, dut.u_unified_buffer.base_row
            );
        end
`endif
        if (cycle_ctr > TB_TOTAL_CYCLE_WATCHDOG) begin
            $fatal(1, "tb_mnist_utpu_program watchdog tripped");
        end
    end

    initial begin
`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_mnist_utpu_program.vcd");
        $dumpvars(0, tb_mnist_utpu_program);
`endif
        trace_fd = $fopen("build/reports/rtl_mnist_utpu_trace.log", "w");
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
        case1_actual_n = 0;
        case1_passed = 1'b0;

        case2_expected[0] = `CASE2_EXP_BYTE_0;
        case2_expected[1] = `CASE2_EXP_BYTE_1;
        case2_expected[2] = `CASE2_EXP_BYTE_2;
        case2_expected[3] = `CASE2_EXP_BYTE_3;
        case2_expected[4] = `CASE2_EXP_BYTE_4;
        case2_expected[5] = `CASE2_EXP_BYTE_5;
        case2_actual_n = 0;
        case2_passed = 1'b0;

        case3_expected[0] = `CASE3_EXP_BYTE_0;
        case3_expected[1] = `CASE3_EXP_BYTE_1;
        case3_expected[2] = `CASE3_EXP_BYTE_2;
        case3_expected[3] = `CASE3_EXP_BYTE_3;
        case3_expected[4] = `CASE3_EXP_BYTE_4;
        case3_expected[5] = `CASE3_EXP_BYTE_5;
        case3_actual_n = 0;
        case3_passed = 1'b0;

        $display("=================================================");
        $display("MNIST 8x8 uTPU demo RTL testbench");
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        run_case(`CASE1_NAME, `CASE1_MEM, `CASE1_WORDS, `CASE1_FETCH_N, case1_actual_n, case1_passed);
        if (TB_CASE_LIMIT >= 2) run_case(`CASE2_NAME, `CASE2_MEM, `CASE2_WORDS, `CASE2_FETCH_N, case2_actual_n, case2_passed);
        if (TB_CASE_LIMIT >= 3) run_case(`CASE3_NAME, `CASE3_MEM, `CASE3_WORDS, `CASE3_FETCH_N, case3_actual_n, case3_passed);

        $display("CASE1_PASS=%0d", case1_passed);
        $display("CASE2_PASS=%0d", case2_passed);
        $display("CASE3_PASS=%0d", case3_passed);
        $display("FIRST_FAILURE_STAGE=%0d", first_failure_stage);
        $display("FIRST_FAILURE_CYCLE=%0d", first_failure_cycle);
        $display("FIRST_FAILURE_INSTRUCTION=%04h", first_failure_instruction);
        $display("TOTAL_CYCLES=%0d", cycle_ctr);
        $display("TRACE_LOG_PATH=build/reports/rtl_mnist_utpu_trace.log");

        $display("=================================================");
        $display("DONE tests=%0d errors=%0d", tests, errors);
        $display("=================================================");
        if (errors != 0) begin
            $fclose(trace_fd);
            $fatal(1, "tb_mnist_utpu_program FAILED");
        end
        $fclose(trace_fd);
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
