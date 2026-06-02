`timescale 1ns/1ps
`include "build/test_vectors/latency_expected.svh"

// -----------------------------------------------------------------------------
// Task 4 — deterministic-latency RTL data-independence testbench.
//
// Loads one compiled uTPU ISA program (path + word-count provided by
// the generated header ``build/test_vectors/latency_expected.svh``),
// streams it through the UART upload protocol, sends MAGIC_START,
// and counts RTL clock cycles between MAGIC_START and HALT_STATE.
//
// The testbench prints exactly one line that the host harness parses:
//
//     LATENCY_CYCLES=<int>
//
// This testbench is intentionally simpler than
// ``tb_scheduler_cycles.sv``: it runs a SINGLE program (not a
// naive-vs-scheduled pair), records its RTL cycle count, and exits.
// The host harness (``firmware/host/run_latency_determinism.py``)
// invokes this same testbench multiple times — once per (shape ×
// input-distribution) — and asserts the cycle count is invariant
// across distributions for the same shape. That invariance, together
// with the static-cycle prover in
// ``firmware/host/latency_analysis.py``, is the empirical
// data-independence guarantee Task 4 delivers.
//
// The testbench does NOT cross-check absolute static-vs-RTL cycle
// counts (Phase 7 remediation P4.1 has already established that the
// simulator's 1-cycle-per-op accounting differs from the RTL FSM's
// multi-cycle STORE/FETCH paths at the absolute count level; they
// agree only at the ±2.0% percentage-cycle-reduction level). The
// RTL cycle count this testbench emits is recorded in the artifact
// as ``rtl_cycles_observed`` and used purely as a data-independence
// witness (variance across distributions, not an absolute equality
// check vs the static model).
// -----------------------------------------------------------------------------

module tb_latency_determinism;
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
`ifdef LATENCY_TB_PROG_DEPTH
    localparam int TB_PROG_DEPTH = `LATENCY_TB_PROG_DEPTH;
`else
    localparam int TB_PROG_DEPTH = 1024;
`endif
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [7:0] MAGIC_REARM  = 8'hA3;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .PROG_DEPTH(TB_PROG_DEPTH)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    int tests = 0;
    int errors = 0;
    longint cycle_ctr = 0;

    reg [15:0] case_mem [0:TB_PROG_DEPTH-1];

    longint start_cycle = 0;
    longint halt_cycle = 0;
    longint observed_cycles = 0;

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
        while (dut.fifo_in.full) begin
            @(posedge clk);
        end
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

    task automatic wait_for_state(input logic [4:0] state_val, input int max_cycles, output bit ok);
        int i;
        ok = 1'b0;
        i = 0;
        while (i < max_cycles && !ok) begin
            @(posedge clk);
            if (dut.current_state == state_val) begin
                ok = 1'b1;
            end
            i = i + 1;
        end
    endtask

    initial begin
        bit reached_wait_start;
        bit reached_halt;
        int slack;

`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_latency_determinism.vcd");
        $dumpvars(0, tb_latency_determinism);
`endif

        $display("=================================================");
        $display("Task 4 -- latency determinism RTL trial");
        $display("program=%s words=%0d shape=%s distribution=%s",
                 `LATENCY_MEM, `LATENCY_WORDS, `LATENCY_SHAPE_TAG, `LATENCY_DIST_TAG);
        $display("=================================================");

        for (int idx = 0; idx < TB_PROG_DEPTH; idx++) case_mem[idx] = 16'h0000;
        $readmemh(`LATENCY_MEM, case_mem);

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        stream_program(`LATENCY_WORDS);
        wait_for_state(dut.WAIT_START_STATE, 500000, reached_wait_start);
        CHECK("reached WAIT_START", reached_wait_start);

        @(posedge clk);
        start_cycle = cycle_ctr;
        push_rx_byte(MAGIC_START);

        slack = 0;
        reached_halt = 1'b0;
        while (!reached_halt && slack < 2_000_000) begin
            @(posedge clk);
            if (dut.current_state == dut.HALT_STATE) reached_halt = 1'b1;
            slack = slack + 1;
        end
        halt_cycle = cycle_ctr;
        CHECK("reached HALT", reached_halt);

        observed_cycles = halt_cycle - start_cycle;
        $display("LATENCY_CYCLES=%0d SHAPE=%s DISTRIBUTION=%s WORDS=%0d",
                 observed_cycles, `LATENCY_SHAPE_TAG, `LATENCY_DIST_TAG, `LATENCY_WORDS);

        $display("=================================================");
        $display("DONE tests=%0d errors=%0d", tests, errors);
        $display("=================================================");

        if (errors != 0) begin
            $fatal(1, "tb_latency_determinism FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end

    always @(posedge clk) cycle_ctr <= cycle_ctr + 1;
endmodule
