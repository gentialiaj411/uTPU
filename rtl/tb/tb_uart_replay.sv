`timescale 1ns/1ps
`include "build/test_vectors/uart_replay_expected.svh"

module tb_uart_replay;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

`ifdef UART_REPLAY_CLK_PERIOD_NS
    localparam time CLK_PERIOD = `UART_REPLAY_CLK_PERIOD_NS * 1ns;
`else
    localparam time CLK_PERIOD = 10ns;
`endif
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = `UART_REPLAY_ARRAY_SIZE;
    localparam int TB_BUFFER_SIZE = `UART_REPLAY_BUFFER_SIZE;
    localparam int TB_PROG_DEPTH = `UART_REPLAY_PROG_DEPTH;
    localparam int TB_FIFO_WIDTH = 2048;
    localparam int TB_EXT_ADDR_EN = `UART_REPLAY_EXT_ADDR_EN;
    localparam int TB_COMPUTE_DATA_WIDTH = `UART_REPLAY_COMPUTE_DATA_WIDTH;
    localparam int TB_ACCUMULATOR_DATA_WIDTH = `UART_REPLAY_ACCUMULATOR_DATA_WIDTH;
    localparam int TB_UART_BAUD = `UART_REPLAY_UART_BAUD;
`ifdef UART_REPLAY_CORE_CLK_HZ
    localparam int TB_CORE_CLK_HZ = `UART_REPLAY_CORE_CLK_HZ;
`else
    localparam int TB_CORE_CLK_HZ = 100000000;
`endif
`ifdef UART_REPLAY_RST_ACTIVE_LOW
    localparam int TB_RST_ACTIVE_LOW = `UART_REPLAY_RST_ACTIVE_LOW;
`else
    localparam int TB_RST_ACTIVE_LOW = 1;
`endif
`ifdef UART_REPLAY_MAX_BATCH_COUNT
    localparam int TB_MAX_BATCH_COUNT = `UART_REPLAY_MAX_BATCH_COUNT;
`else
    localparam int TB_MAX_BATCH_COUNT = 64;
`endif
    localparam int OVERSAMPLE = 16;
    localparam int BIT_CYCLES = 16;
    localparam int RESULT_WAIT_CYCLES = 500000;
    localparam int WAIT_START_CYCLES = 500000;
    localparam logic [7:0] MAGIC_START = 8'hA2;
    localparam logic [7:0] MAGIC_REPEAT = 8'hA6;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
`ifdef UART_REPLAY_WEIGHT_REGION_WORDS
        .WEIGHT_REGION_WORDS(`UART_REPLAY_WEIGHT_REGION_WORDS),
`endif
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .PROG_DEPTH(TB_PROG_DEPTH),
        .EXT_ADDR_EN(TB_EXT_ADDR_EN),
        .COMPUTE_DATA_WIDTH(TB_COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(TB_ACCUMULATOR_DATA_WIDTH),
        .MAX_BATCH_COUNT(TB_MAX_BATCH_COUNT),
        .UART_INPUT_CLK(TB_CORE_CLK_HZ),
        .RST_ACTIVE_LOW(TB_RST_ACTIVE_LOW),
        .UART_BAUD(TB_UART_BAUD)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    logic tx_mon_valid;
    logic [7:0] tx_mon_byte;
    uart_receiver #(
        .UART_BITS_TRANSFERED(8),
        .OVERSAMPLE(OVERSAMPLE)
    ) tx_monitor (
        .clk(clk),
        .rst(dut.rst_int),
        .baud_tick(dut.u_uart.baud_tick),
        .rx(tx),
        .valid(tx_mon_valid),
        .result(tx_mon_byte)
    );

    int tests = 0;
    int errors = 0;
    reg [7:0] upload_mem [0:`UART_REPLAY_UPLOAD_N-1];
    reg [7:0] expected_mem [0:`UART_REPLAY_EXPECTED_N-1];
    reg [7:0] actual_mem [0:`UART_REPLAY_EXPECTED_N-1];
    logic quantizer_x_seen;
    logic uart_tx_x_seen;
    logic capture_enabled;

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

    task automatic push_serial_byte(input logic [7:0] b);
        int bit_idx;
        begin
            rx <= 1'b0;
            wait_cycles(BIT_CYCLES);
            for (bit_idx = 0; bit_idx < 8; bit_idx++) begin
                rx <= b[bit_idx];
                wait_cycles(BIT_CYCLES);
            end
            rx <= 1'b1;
            wait_cycles(BIT_CYCLES);
        end
    endtask

    task automatic wait_for_state(input logic [4:0] state_val, input int max_cycles, output bit ok);
        int i;
        begin
            ok = 1'b0;
            i = 0;
            while (i < max_cycles && !ok) begin
                @(posedge clk);
                if (dut.current_state == state_val)
                    ok = 1'b1;
                i = i + 1;
            end
        end
    endtask

    initial begin
        bit reached_wait_start;
        bit reached_halt;
        int i;
        int got;
        int spins;

        for (i = 0; i < `UART_REPLAY_EXPECTED_N; i++) actual_mem[i] = 8'h00;
        $readmemh(`UART_REPLAY_UPLOAD_MEM, upload_mem);
        $readmemh(`UART_REPLAY_EXPECTED_MEM, expected_mem);
        quantizer_x_seen = 1'b0;
        uart_tx_x_seen = 1'b0;
        capture_enabled = 1'b0;

        rst <= TB_RST_ACTIVE_LOW ? 1'b0 : 1'b1;
        rx <= 1'b1;
        wait_cycles(10);
        rst <= TB_RST_ACTIVE_LOW ? 1'b1 : 1'b0;
        wait_cycles(40);

        for (i = 0; i < `UART_REPLAY_UPLOAD_N; i++) begin
            push_serial_byte(upload_mem[i]);
        end

        wait_for_state(dut.WAIT_START_STATE, WAIT_START_CYCLES, reached_wait_start);
        CHECK("Reached WAIT_START after serial upload", reached_wait_start);

        capture_enabled <= 1'b1;
        push_serial_byte(MAGIC_START);

        got = 0;
        spins = 0;
        while (got < `UART_REPLAY_EXPECTED_N && spins < RESULT_WAIT_CYCLES) begin
            @(posedge clk);
            spins = spins + 1;
            if (capture_enabled && tx === 1'bx)
                uart_tx_x_seen = 1'b1;
            if (dut.requant_finalize_enable && dut.writeback_wait_clear) begin
                for (i = 0; i < dut.QUANTIZER_SIZE; i++) begin
                    if ((^dut.quantizer_out[i]) === 1'bx)
                        quantizer_x_seen = 1'b1;
                end
            end
            if (tx_mon_valid) begin
                if ((got == 0) && (tx_mon_byte == 8'hAA)) begin
                    // ignore reset self-test byte if it arrives late
                end else if (got < `UART_REPLAY_EXPECTED_N) begin
                    actual_mem[got] = tx_mon_byte;
                    got = got + 1;
                end
            end
        end

        wait_for_state(dut.HALT_STATE, WAIT_START_CYCLES, reached_halt);
        CHECK("Reached HALT after UART start", reached_halt);
        CHECK("Captured expected UART byte count", got == `UART_REPLAY_EXPECTED_N);
        CHECK("No X on quantizer finalize outputs", !quantizer_x_seen);
        CHECK("No X on UART TX line during capture", !uart_tx_x_seen);
        for (i = 0; i < `UART_REPLAY_EXPECTED_N; i++) begin
            if (actual_mem[i] !== expected_mem[i])
                $display("UART_BYTE_MISMATCH idx=%0d expected=%02x actual=%02x", i, expected_mem[i], actual_mem[i]);
            CHECK($sformatf("UART byte %0d", i), actual_mem[i] === expected_mem[i]);
        end

        // Buffer-resident multi-inference deployments upload control once.
        // A6 returns HALT -> WAIT_START and restarts the retained program at PC=0.
        push_serial_byte(MAGIC_REPEAT);
        wait_for_state(dut.WAIT_START_STATE, WAIT_START_CYCLES, reached_wait_start);
        CHECK("A6 repeat retains uploaded program", reached_wait_start);
        push_serial_byte(MAGIC_START);
        got = 0;
        spins = 0;
        while (got < `UART_REPLAY_EXPECTED_N && spins < RESULT_WAIT_CYCLES) begin
            @(posedge clk);
            spins = spins + 1;
            if (tx_mon_valid && got < `UART_REPLAY_EXPECTED_N) begin
                actual_mem[got] = tx_mon_byte;
                got = got + 1;
            end
        end
        wait_for_state(dut.HALT_STATE, WAIT_START_CYCLES, reached_halt);
        CHECK("A6 repeated execution reaches HALT", reached_halt);
        CHECK("A6 repeated execution emits full output", got == `UART_REPLAY_EXPECTED_N);
        for (i = 0; i < `UART_REPLAY_EXPECTED_N; i++)
            CHECK($sformatf("A6 repeated UART byte %0d", i), actual_mem[i] === expected_mem[i]);
        $write("UART_BYTES_ACTUAL=");
        for (i = 0; i < `UART_REPLAY_EXPECTED_N; i++) begin
            $write("%02x", actual_mem[i]);
            if (i + 1 < `UART_REPLAY_EXPECTED_N)
                $write(",");
        end
        $write("\n");
        $display("PERF_CYCLE_COUNTER=%0d", dut.perf_cycle_counter);
        $display("PERF_BUSY_COUNTER=%0d", dut.perf_busy_counter);
        $display("PERF_PROGRAM_COUNT=%0d", dut.perf_program_count);
        $display("UART_REPLAY_PROGRAM_WORDS=%0d", `UART_REPLAY_WORDS);
        $display("UART_REPLAY_PROG_DEPTH=%0d", `UART_REPLAY_PROG_DEPTH);

        if (errors != 0) begin
            $display("TB_RESULT: FAIL");
            $fatal(1, "tb_uart_replay FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
