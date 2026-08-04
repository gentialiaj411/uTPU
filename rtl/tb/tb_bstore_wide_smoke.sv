`timescale 1ns/1ps

// BSTORE write-arm smoke: EXT_ADDR burst of 16 words with BSTORE_WIDTH=8.
// Checks buffer contents and that attr cycles/word drop below the legacy 4.0.
module tb_bstore_wide_smoke;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = 8;
    localparam int TB_BUFFER_SIZE = 4096;
    localparam int TB_COMPUTE_DW = 8;
    localparam int TB_ACC_DW = 32;
    localparam int TB_FIFO_WIDTH = 256;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);
    localparam int TB_BSTORE_WIDTH = 8;
    localparam int TB_COUNT = 16;

    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [15:0] W_BSTORE = 16'h0006; // OPCODE_BSTORE
    localparam logic [15:0] W_ADDR   = 16'h0000;
    localparam logic [15:0] W_COUNT  = TB_COUNT;
    localparam logic [15:0] W_HALT   = 16'h0004;
    // header + addr + count + payloads + halt
    localparam int PROG_WORDS = 3 + TB_COUNT + 1;

    int tests = 0;
    int errors = 0;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .COMPUTE_DATA_WIDTH(TB_COMPUTE_DW),
        .ACCUMULATOR_DATA_WIDTH(TB_ACC_DW),
        .EXT_ADDR_EN(1),
        .PROG_DEPTH(8192),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .MAX_BATCH_COUNT(4),
        .QUANTIZER_PIPE_DEPTH(1),
        .BSTORE_WIDTH(TB_BSTORE_WIDTH)
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

    task automatic push_word(input logic [15:0] w);
        push_rx_byte(w[7:0]);
        push_rx_byte(w[15:8]);
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

    function automatic logic [15:0] peek_buffer_word(input int addr);
        int banks;
        int bank;
        int row;
        banks = dut.u_unified_buffer.BANKS;
        bank = addr % banks;
        row = addr / banks;
        return dut.u_unified_buffer.bank_mem[bank][row];
    endfunction

    initial begin
        bit reached_wait_start;
        bit reached_halt;
        logic [15:0] got;
        real cyc_per_word;
        int i;

`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_bstore_wide_smoke.vcd");
        $dumpvars(0, tb_bstore_wide_smoke);
`endif

        $display("=================================================");
        $display("BSTORE wide smoke  WIDTH=%0d COUNT=%0d", TB_BSTORE_WIDTH, TB_COUNT);
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        CHECK("BSTORE_WIDTH==8", dut.BSTORE_WIDTH == TB_BSTORE_WIDTH);

        push_rx_byte(MAGIC_UPLOAD);
        push_rx_byte(PROG_WORDS[7:0]);
        push_rx_byte(PROG_WORDS[15:8]);
        push_word(W_BSTORE);
        push_word(W_ADDR);
        push_word(W_COUNT);
        for (i = 0; i < TB_COUNT; i++)
            push_word(16'h1000 + i[15:0]);
        push_word(W_HALT);

        wait_for_state(dut.WAIT_START_STATE, 100000, reached_wait_start);
        CHECK("Reached WAIT_START after upload", reached_wait_start);
        wait_cycles(2);

        push_rx_byte(MAGIC_START);
        wait_for_state(dut.HALT_STATE, 200000, reached_halt);
        CHECK("Reached HALT after start", reached_halt);

        for (i = 0; i < TB_COUNT; i++) begin
            got = peek_buffer_word(i);
            if (got !== (16'h1000 + i[15:0]))
                $display("[DBG] buffer[%0d] got=0x%04x expect=0x%04x",
                         i, got, 16'h1000 + i[15:0]);
            CHECK($sformatf("buffer[%0d]==0x%04x", i, 16'h1000 + i[15:0]),
                  got === (16'h1000 + i[15:0]));
        end

        cyc_per_word = real'(dut.perf_attr_bstore) / real'(TB_COUNT);
        $display("ATTR_BSTORE=%0d cyc_per_word=%0f", dut.perf_attr_bstore, cyc_per_word);
        CHECK("cyc_per_word < 2.0 (legacy was 4.0)", cyc_per_word < 2.0);
        CHECK("cyc_per_word > 0", dut.perf_attr_bstore > 0);

        if (errors == 0)
            $display("BSTORE_WIDE_SMOKE_PASS tests=%0d attr=%0d cyc_per_word=%0f",
                     tests, dut.perf_attr_bstore, cyc_per_word);
        else
            $display("BSTORE_WIDE_SMOKE_FAIL errors=%0d/%0d", errors, tests);
        $finish;
    end
endmodule
