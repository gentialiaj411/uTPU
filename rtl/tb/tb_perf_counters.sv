`timescale 1ns/1ps

module tb_perf_counters;
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
    localparam logic [7:0] MAGIC_START = 8'hA2;
    localparam logic [7:0] MAGIC_READ_PERF = 8'hA4;

    int tests = 0;
    int errors = 0;
    byte perf_bytes [0:23];
    logic [63:0] cycle_ctr;
    logic [63:0] busy_ctr;
    logic [63:0] program_ctr;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .FIFO_WIDTH(TB_FIFO_WIDTH)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
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

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    task automatic push_rx_byte(input logic [7:0] b);
        @(posedge clk);
        dut.fifo_in.mem[dut.fifo_in.w_ptr[TB_FIFO_PTR_W-1:0]] = b;
        dut.fifo_in.w_ptr = dut.fifo_in.w_ptr + 1'b1;
        @(posedge clk);
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

    task automatic collect_perf_bytes(output int got);
        int i;
        got = 0;
        i = 0;
        while (i < 100000 && got < 24) begin
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
        int got_perf;
        int i;

`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_perf_counters.vcd");
        $dumpvars(0, tb_perf_counters);
`endif

        $display("=================================================");
        $display("perf counter RTL testbench");
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        // Upload a one-word HALT program.
        push_rx_byte(MAGIC_UPLOAD);
        push_rx_byte(8'h01); // length low byte (1 word)
        push_rx_byte(8'h00); // length high byte
        push_rx_byte(8'h04); // HALT opcode (low byte)
        push_rx_byte(8'h00); // high byte

        wait_for_state(dut.WAIT_START_STATE, 20000, reached_wait_start);
        CHECK("Reached WAIT_START after upload", reached_wait_start);

        // Start execution; HALT should increment program count once.
        push_rx_byte(MAGIC_START);
        wait_for_state(dut.HALT_STATE, 20000, reached_halt);
        CHECK("Reached HALT after start", reached_halt);

        // Request perf counters and collect 24-byte response.
        push_rx_byte(MAGIC_READ_PERF);
        collect_perf_bytes(got_perf);
        CHECK("Received 24 perf bytes", got_perf == 24);

        cycle_ctr = '0;
        busy_ctr = '0;
        program_ctr = '0;
        for (i = 0; i < 8; i++) cycle_ctr = {cycle_ctr[55:0], perf_bytes[i]};
        for (i = 8; i < 16; i++) busy_ctr = {busy_ctr[55:0], perf_bytes[i]};
        for (i = 16; i < 24; i++) program_ctr = {program_ctr[55:0], perf_bytes[i]};

        CHECK("Cycle counter advanced", cycle_ctr > 64'd0);
        CHECK("Busy counter bounded by cycle counter", busy_ctr <= cycle_ctr);
        CHECK("Program count incremented on HALT", program_ctr >= 64'd1);

        $display("PERF_CYCLE_COUNTER=%0d", cycle_ctr);
        $display("PERF_BUSY_COUNTER=%0d", busy_ctr);
        $display("PERF_PROGRAM_COUNT=%0d", program_ctr);
        $display("=================================================");
        $display("DONE tests=%0d errors=%0d", tests, errors);
        $display("=================================================");

        if (errors != 0) begin
            $fatal(1, "tb_perf_counters FAILED");
        end
        $display("TB_RESULT: PASS");
        $finish;
    end
endmodule
