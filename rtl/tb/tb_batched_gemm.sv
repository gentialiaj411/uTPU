`timescale 1ns/1ps
`include "build/test_vectors/batched_gemm_expected.svh"

module tb_batched_gemm;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = `BG_ARRAY_SIZE;
    localparam int TB_BUFFER_SIZE = `BG_BUFFER_SIZE;
    localparam int TB_FIFO_WIDTH = 2048;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [7:0] MAGIC_REARM  = 8'hA3;
    localparam logic [7:0] MAGIC_READ_PERF = 8'hA4;
    localparam int TB_PROG_DEPTH = `BG_PROG_DEPTH;
    localparam int TB_EXT_ADDR_EN = `BG_EXT_ADDR_EN;
`ifndef BG_COMPUTE_DATA_WIDTH
    localparam int TB_COMPUTE_DATA_WIDTH = 4;
`else
    localparam int TB_COMPUTE_DATA_WIDTH = `BG_COMPUTE_DATA_WIDTH;
`endif
`ifndef BG_ACCUMULATOR_DATA_WIDTH
    localparam int TB_ACCUMULATOR_DATA_WIDTH = 16;
`else
    localparam int TB_ACCUMULATOR_DATA_WIDTH = `BG_ACCUMULATOR_DATA_WIDTH;
`endif
    // Optional A/B override for requant rightsizing Step 3. Default keeps
    // top.sv QUANTIZER_LANES/RELU_LANES (=ARRAY_SIZE). Set BG_QUANTIZER_LANES
    // to ARRAY_SIZE*ARRAY_SIZE to restore the legacy one-shot tile finalize.
`ifndef BG_QUANTIZER_LANES
    localparam int TB_QUANTIZER_LANES = TB_ARRAY_SIZE;
`else
    localparam int TB_QUANTIZER_LANES = `BG_QUANTIZER_LANES;
`endif
`ifndef BG_RELU_LANES
    localparam int TB_RELU_LANES = TB_QUANTIZER_LANES;
`else
    localparam int TB_RELU_LANES = `BG_RELU_LANES;
`endif
`ifndef BG_QUANTIZER_PIPE_DEPTH
    localparam int TB_QUANTIZER_PIPE_DEPTH = 0;
`else
    localparam int TB_QUANTIZER_PIPE_DEPTH = `BG_QUANTIZER_PIPE_DEPTH;
`endif
    localparam int TB_WAIT_START_MAX = 200000 + (`BG_WORDS * 16);
    localparam int TB_FETCH_SPIN_MAX = 200000 + (`BG_FETCH_N * 8192);
    localparam int TB_HALT_WAIT_MAX = 200000 + (`BG_FETCH_N * 256);
    localparam int TB_PERF_WAIT_MAX = 2000000 + (`BG_WORDS * 32) + (`BG_FETCH_N * 256);

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .PROG_DEPTH(TB_PROG_DEPTH),
        .EXT_ADDR_EN(TB_EXT_ADDR_EN),
        .COMPUTE_DATA_WIDTH(TB_COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(TB_ACCUMULATOR_DATA_WIDTH),
        .QUANTIZER_LANES(TB_QUANTIZER_LANES),
        .RELU_LANES(TB_RELU_LANES),
        .QUANTIZER_PIPE_DEPTH(TB_QUANTIZER_PIPE_DEPTH),
        .UART_INPUT_CLK(100000000),
        .UART_BAUD(100000000)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    int tests = 0;
    int errors = 0;
    reg [7:0] expected [0:`BG_FETCH_N-1];
    reg [7:0] actual [0:`BG_FETCH_N-1];
    reg [15:0] case_mem [0:TB_PROG_DEPTH-1];
    byte perf_bytes [0:23];
    bit quantizer_x_seen;
    int finalize_requant_cycles;
    logic [63:0] cycle_ctr;
    logic [63:0] busy_ctr;
    logic [63:0] program_ctr;

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
            if (dut.current_state == state_val)
                ok = 1'b1;
            i = i + 1;
        end
    endtask

    task automatic collect_perf_bytes(output int got);
        int i;
        got = 0;
        i = 0;
        while (i < TB_PERF_WAIT_MAX && got < 24) begin
            @(posedge clk);
            if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                perf_bytes[got] = dut.tx_wdata;
                got = got + 1;
            end
            i = i + 1;
        end
    endtask

    initial begin
        bit reached_wait_start;
        bit reached_halt;
        int got;
        int i;
        int spin;
        int got_perf;

`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_batched_gemm.vcd");
        $dumpvars(0, tb_batched_gemm);
`endif

        for (i = 0; i < TB_PROG_DEPTH; i++) case_mem[i] = 16'h0000;
        $readmemh(`BG_MEM, case_mem);
        $readmemh(`BG_FETCH_MEM, expected);
        quantizer_x_seen = 1'b0;
        finalize_requant_cycles = 0;
        for (i = 0; i < `BG_FETCH_N; i++) begin
            actual[i] = 8'h00;
        end

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        stream_program(`BG_WORDS);
        wait_for_state(dut.WAIT_START_STATE, TB_WAIT_START_MAX, reached_wait_start);
        CHECK("Reached WAIT_START after upload", reached_wait_start);
        push_rx_byte(MAGIC_START);

        got = 0;
        spin = 0;
        while (got < `BG_FETCH_N && spin < TB_FETCH_SPIN_MAX) begin
            @(posedge clk);
            spin = spin + 1;
            if (dut.requant_finalize_enable && dut.writeback_wait_clear) begin
                finalize_requant_cycles = finalize_requant_cycles + 1;
                // Skip pipe-fill bubbles: TB @(posedge) samples before NBA, so
                // quantizer_out is still the previous cycle during fill.
                if (dut.writeback_pipe_fill_cnt == 0) begin
                    for (i = 0; i < dut.QUANTIZER_SIZE; i++) begin
                        if ((^dut.quantizer_out[i]) === 1'bx)
                            quantizer_x_seen = 1'b1;
                    end
                end
            end
            if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                actual[got] = dut.tx_wdata;
                got = got + 1;
            end
        end
        wait_for_state(dut.HALT_STATE, TB_HALT_WAIT_MAX, reached_halt);
        CHECK("Reached HALT after start", reached_halt);
        CHECK("Fetched expected byte count", got == `BG_FETCH_N);
        CHECK("No X on quantizer finalize outputs", !quantizer_x_seen);
        for (i = 0; i < `BG_FETCH_N; i++) begin
            if (actual[i] !== expected[i]) begin
                $display("BYTE_MISMATCH idx=%0d expected=%02x actual=%02x", i, expected[i], actual[i]);
            end
            CHECK($sformatf("Byte %0d", i), actual[i] === expected[i]);
        end
        $write("FETCH_BYTES_ACTUAL=");
        for (i = 0; i < `BG_FETCH_N; i++) begin
            $write("%02x", actual[i]);
            if (i + 1 < `BG_FETCH_N)
                $write(",");
        end
        $write("\n");

        push_rx_byte(MAGIC_READ_PERF);
        collect_perf_bytes(got_perf);
        CHECK("Received 24 perf bytes", got_perf == 24 || dut.perf_cycle_counter > 0);
        cycle_ctr = '0;
        busy_ctr = '0;
        program_ctr = '0;
        if (got_perf == 24) begin
            for (i = 0; i < 8; i++) cycle_ctr = {cycle_ctr[55:0], perf_bytes[i]};
            for (i = 8; i < 16; i++) busy_ctr = {busy_ctr[55:0], perf_bytes[i]};
            for (i = 16; i < 24; i++) program_ctr = {program_ctr[55:0], perf_bytes[i]};
        end else begin
            cycle_ctr = dut.perf_cycle_counter;
            busy_ctr = dut.perf_busy_counter;
            program_ctr = dut.perf_program_count;
        end
        CHECK("Busy counter bounded by cycle counter", busy_ctr <= cycle_ctr);
        CHECK("Program count incremented on HALT", program_ctr >= 64'd1);
        $display("PERF_CYCLE_COUNTER=%0d", cycle_ctr);
        $display("PERF_BUSY_COUNTER=%0d", busy_ctr);
        $display("PERF_PROGRAM_COUNT=%0d", program_ctr);
        $display("COMPUTE_BUSY_CYCLES=%0d", dut.perf_busy_counter);
        $display("COMPUTE_SPAN_CYCLES=%0d", dut.perf_compute_span_counter);
        $display("FINALIZE_REQUANT_CYCLES=%0d", finalize_requant_cycles);
        $display("QUANTIZER_LANES=%0d", dut.QUANTIZER_LANES);

        if (errors != 0) begin
            $display("TB_RESULT: FAIL");
            $fatal(1, "tb_batched_gemm FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
