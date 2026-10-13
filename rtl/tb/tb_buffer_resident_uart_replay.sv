`timescale 1ns/1ps
`include "build/test_vectors/buffer_resident_uart_replay.svh"

module tb_buffer_resident_uart_replay;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;
    localparam time CLK_PERIOD = `UART_REPLAY_CLK_PERIOD_NS * 1ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int BIT_CYCLES = 16;
    localparam int MAX_WAIT = 2000000;
    localparam int ARRAY_SIZE = `UART_REPLAY_ARRAY_SIZE;
    localparam int BUFFER_SIZE = `UART_REPLAY_BUFFER_SIZE;
    localparam int PROG_DEPTH = `UART_REPLAY_PROG_DEPTH;
    localparam int UART_BAUD = `UART_REPLAY_UART_BAUD;
    localparam int CORE_CLK_HZ = `UART_REPLAY_CORE_CLK_HZ;
    localparam int RST_ACTIVE_LOW = `UART_REPLAY_RST_ACTIVE_LOW;
    localparam int MAX_BATCH_COUNT = `UART_REPLAY_MAX_BATCH_COUNT;

    top #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .BUFFER_SIZE(BUFFER_SIZE),
        .WEIGHT_REGION_WORDS(`UART_REPLAY_WEIGHT_REGION_WORDS),
        .PROG_DEPTH(PROG_DEPTH),
        .EXT_ADDR_EN(1),
        .COMPUTE_DATA_WIDTH(8),
        .ACCUMULATOR_DATA_WIDTH(32),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .UART_INPUT_CLK(CORE_CLK_HZ),
        .RST_ACTIVE_LOW(RST_ACTIVE_LOW),
        .UART_BAUD(UART_BAUD)
    ) dut (.clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst));

    logic tx_mon_valid;
    logic [7:0] tx_mon_byte;
    uart_receiver #(.UART_BITS_TRANSFERED(8), .OVERSAMPLE(16)) tx_monitor (
        .clk(clk), .rst(dut.rst_int), .baud_tick(dut.u_uart.baud_tick),
        .rx(tx), .valid(tx_mon_valid), .result(tx_mon_byte)
    );

    reg [7:0] initial_mem [0:`UART_REPLAY_INITIAL_N-1];
    reg [7:0] inputs_mem [0:`UART_REPLAY_INPUT_N-1];
    reg [7:0] expected_mem [0:`UART_REPLAY_EXPECTED_N-1];
    bit capture_enabled;
    integer global_core_cycles = 0;
    integer cold_start_cycle = 0;
    integer steady_start_cycle = 0;
    always @(posedge clk) global_core_cycles <= global_core_cycles + 1;

    task automatic wait_cycles(input int count);
        repeat (count) @(posedge clk);
    endtask

    task automatic push_serial_byte(input logic [7:0] value);
        int b;
        begin
            rx <= 1'b0;
            wait_cycles(BIT_CYCLES);
            for (b = 0; b < 8; b++) begin
                rx <= value[b];
                wait_cycles(BIT_CYCLES);
            end
            rx <= 1'b1;
            wait_cycles(BIT_CYCLES);
        end
    endtask

    task automatic wait_state(input logic [4:0] target, output bit ok);
        int i;
        begin
            ok = 1'b0;
            for (i = 0; i < MAX_WAIT && !ok; i++) begin
                @(posedge clk);
                if (dut.current_state == target) ok = 1'b1;
            end
        end
    endtask

    initial begin
        bit ok;
        int i, sample, got, spins, errors;
        for (i = 0; i < `UART_REPLAY_EXPECTED_N; i++) expected_mem[i] = 0;
        $readmemh(`UART_REPLAY_INITIAL_MEM, initial_mem);
        $readmemh(`UART_REPLAY_INPUT_MEM, inputs_mem);
        $readmemh(`UART_REPLAY_EXPECTED_MEM, expected_mem);
        capture_enabled = 1'b0;
        errors = 0;

        rst <= RST_ACTIVE_LOW ? 1'b0 : 1'b1;
        rx <= 1'b1;
        wait_cycles(10);
        rst <= RST_ACTIVE_LOW ? 1'b1 : 1'b0;
        wait_cycles(40);

        cold_start_cycle = global_core_cycles;
        for (i = 0; i < `UART_REPLAY_INITIAL_N; i++) push_serial_byte(initial_mem[i]);
        wait_state(dut.WAIT_START_STATE, ok);
        if (!ok) begin $display("FAIL: initial A5 and A1 uploads did not reach WAIT_START"); errors++; end
        capture_enabled = 1'b1;

        for (sample = 0; sample < `UART_REPLAY_SAMPLES; sample++) begin
            if (sample != 0) begin
                steady_start_cycle = global_core_cycles;
                push_serial_byte(8'hA6); // retain program image, reset PC
                wait_state(dut.WAIT_START_STATE, ok);
                if (!ok) begin $display("FAIL: A6 repeat did not reach WAIT_START sample=%0d", sample); errors++; end
            end
            for (i = 0; i < `UART_REPLAY_INPUT_N_PER_SAMPLE; i++)
                push_serial_byte(inputs_mem[sample*`UART_REPLAY_INPUT_N_PER_SAMPLE+i]);
            push_serial_byte(8'hA2);

            got = 0;
            spins = 0;
            while (got < `UART_REPLAY_OUTPUT_N_PER_SAMPLE && spins < MAX_WAIT) begin
                @(posedge clk);
                spins++;
                if (tx_mon_valid && capture_enabled && got < `UART_REPLAY_OUTPUT_N_PER_SAMPLE) begin
                    if (tx_mon_byte !== expected_mem[sample*`UART_REPLAY_OUTPUT_N_PER_SAMPLE+got]) begin
                        $display("UART_MISMATCH sample=%0d byte=%0d expected=%02x actual=%02x",
                                 sample, got,
                                 expected_mem[sample*`UART_REPLAY_OUTPUT_N_PER_SAMPLE+got], tx_mon_byte);
                        errors++;
                    end
                    got++;
                end
            end
            wait_state(dut.HALT_STATE, ok);
            if (!ok || got != `UART_REPLAY_OUTPUT_N_PER_SAMPLE) begin
                $display("FAIL: sample=%0d reached_halt=%0d bytes=%0d expected=%0d", sample, ok, got,
                         `UART_REPLAY_OUTPUT_N_PER_SAMPLE);
                errors++;
            end
            $display("SAMPLE_PERF_PROGRAM_CYCLES index=%0d cycles=%0d", sample,
                     dut.perf_program_cycle_counter);
            if (sample == 0)
                $display("COLD_CORE_CYCLES_INCLUDING_WEIGHT_PROGRAM_INPUT_FILL=%0d",
                         global_core_cycles - cold_start_cycle);
            else
                $display("STEADY_CORE_CYCLES_INCLUDING_INPUT_FILL index=%0d cycles=%0d",
                         sample, global_core_cycles - steady_start_cycle);
            $display("SAMPLE_RESULT index=%0d byte_count=%0d pass=%0d", sample, got,
                     (ok && got == `UART_REPLAY_OUTPUT_N_PER_SAMPLE));
        end
        $display("PERF_PROGRAM_CYCLES=%0d", dut.perf_program_cycle_counter);
        $display("BUFFER_RESIDENT_REPLAY_SAMPLES=%0d", `UART_REPLAY_SAMPLES);
        if (errors) begin
            $display("TB_RESULT: FAIL errors=%0d", errors);
            $fatal(1, "buffer-resident UART replay failed");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
