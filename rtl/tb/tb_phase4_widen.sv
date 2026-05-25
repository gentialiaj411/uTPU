`timescale 1ns/1ps
`include "build/test_vectors/phase4_widen_expected.svh"

// -----------------------------------------------------------------------------
// Phase 4 widened RTL testbench.
//
// Exercises a single-block INT8 matmul through the parameterised top:
//   COMPUTE_DATA_WIDTH     = 8
//   ACCUMULATOR_DATA_WIDTH = 32
//   ARRAY_SIZE             = 8
//   BUFFER_SIZE            = 4096
//   ADDRESS_SIZE           = 12
//   EXT_ADDR_EN            = 1   (2-word LOAD/RUN/FETCH/BSTORE address layout)
//
// The expected fetch bytes come from ``isa_simulator.py`` (which is itself
// regression-tested against a NumPy oracle in
// ``firmware/host/test_phase4_isa_widen.py``). A pass here means the widened
// RTL drives the same byte sequence the python simulator does at the
// widened ISA configuration. Legacy (unwidened) bitmatch is still covered by
// ``tb_fused_compressed_program.sv``.
// -----------------------------------------------------------------------------

module tb_phase4_widen;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE             = 8;
    localparam int TB_BUFFER_SIZE            = 4096;
    localparam int TB_FIFO_WIDTH             = 256;
    localparam int TB_FIFO_PTR_W             = $clog2(TB_FIFO_WIDTH);
    localparam int TB_COMPUTE_DATA_WIDTH     = 8;
    localparam int TB_ACCUMULATOR_DATA_WIDTH = 32;
    localparam int TB_ADDRESS_SIZE           = $clog2(TB_BUFFER_SIZE);
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [7:0] MAGIC_REARM  = 8'hA3;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .COMPUTE_DATA_WIDTH(TB_COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(TB_ACCUMULATOR_DATA_WIDTH),
        .EXT_ADDR_EN(1)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    int tests = 0;
    int errors = 0;
    int cycle_ctr = 0;

    byte expected_bytes [0:`P4_CASE1_FETCH_N-1];
    byte actual_bytes   [0:`P4_CASE1_FETCH_N-1];
    int actual_n = 0;
    bit halted = 1'b0;

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

    task automatic wait_halt(input int max_cycles, output bit got_halt);
        int i;
        got_halt = 1'b0;
        i = 0;
        while (i < max_cycles && !got_halt) begin
            @(posedge clk);
            if (dut.current_state == dut.HALT_STATE) begin
                got_halt = 1'b1;
            end
            i = i + 1;
        end
    endtask

    initial begin
`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_phase4_widen.vcd");
        $dumpvars(0, tb_phase4_widen);
`endif

        for (int idx = 0; idx < `P4_CASE1_FETCH_N; idx++) begin
            actual_bytes[idx] = 8'h00;
        end
        // Expected bytes from the expected.svh.
        // Generated as: `define P4_CASE1_EXP_BYTE_<idx> 8'h<value>
        `ifdef P4_CASE1_EXP_BYTE_0
            expected_bytes[0] = `P4_CASE1_EXP_BYTE_0;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_1
            expected_bytes[1] = `P4_CASE1_EXP_BYTE_1;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_2
            expected_bytes[2] = `P4_CASE1_EXP_BYTE_2;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_3
            expected_bytes[3] = `P4_CASE1_EXP_BYTE_3;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_4
            expected_bytes[4] = `P4_CASE1_EXP_BYTE_4;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_5
            expected_bytes[5] = `P4_CASE1_EXP_BYTE_5;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_6
            expected_bytes[6] = `P4_CASE1_EXP_BYTE_6;
        `endif
        `ifdef P4_CASE1_EXP_BYTE_7
            expected_bytes[7] = `P4_CASE1_EXP_BYTE_7;
        `endif

        $display("=================================================");
        $display("Phase 4 widened RTL testbench (INT8, ARRAY=8, EXT_ADDR=1)");
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        for (int i = 0; i < 1024; i++) case_mem[i] = 16'h0000;
        $readmemh(`P4_MEM, case_mem);

        // Stream program over the UART RX FIFO and start.
        begin
            bit saw_wait_start;
            stream_program(`P4_WORDS);
            wait_wait_start(200000, saw_wait_start);
            CHECK("phase4 reached WAIT_START", saw_wait_start);
            push_rx_byte(MAGIC_START);
        end

        // Capture transmitted UART bytes (skip the one-shot 0xAA selftest).
        actual_n = 0;
        while (actual_n < `P4_CASE1_FETCH_N && cycle_ctr < 1000000) begin
            @(posedge clk);
            if (dut.tx_we) begin
                if (dut.tx_wdata !== 8'hAA) begin
                    actual_bytes[actual_n] = dut.tx_wdata;
                    actual_n = actual_n + 1;
                end
            end
        end

        wait_halt(50000, halted);
        CHECK("phase4 HALT reached", halted);
        CHECK("phase4 fetch byte count", actual_n == `P4_CASE1_FETCH_N);

        for (int idx = 0; idx < `P4_CASE1_FETCH_N; idx++) begin
            string label;
            label.itoa(idx);
            CHECK({"phase4 byte ", label}, actual_bytes[idx] === expected_bytes[idx]);
            $display("  byte[%0d] expected=0x%02x actual=0x%02x %s",
                     idx, expected_bytes[idx], actual_bytes[idx],
                     actual_bytes[idx] === expected_bytes[idx] ? "" : "<-- MISMATCH");
        end

        $display("=================================================");
        $display("DONE tests=%0d errors=%0d", tests, errors);
        $display("=================================================");

        if (errors != 0) begin
            $fatal(1, "tb_phase4_widen FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end

    always @(posedge clk) cycle_ctr <= cycle_ctr + 1;
endmodule
