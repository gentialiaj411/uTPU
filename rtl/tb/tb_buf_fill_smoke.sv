`timescale 1ns/1ps

// A5 buffer-fill smoke: MAGIC_BUF_FILL writes words, then STORE-free START/HALT
// is not required — just fill + peek buffer. Also checks partition constants.
module tb_buf_fill_smoke;
    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    localparam int TB_ARRAY_SIZE = 8;
    localparam int TB_BUFFER_SIZE = 16384;
    localparam int TB_COMPUTE_DW = 8;
    localparam int TB_ACC_DW = 32;
    localparam int TB_FIFO_WIDTH = 256;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);
    localparam int TB_COUNT = 8;
    localparam int TB_DEST = 100;

    localparam logic [7:0] MAGIC_BUF_FILL = 8'hA5;
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;
    localparam logic [15:0] W_HALT = 16'h0004;

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
        .BUF_FILL_EN(1),
        .QUANTIZER_PIPE_DEPTH(1)
    ) dut (
        .clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst)
    );

    task automatic CHECK(input string name, input bit cond);
        tests++;
        if (!cond) begin
            errors++;
            $display("[FAIL] %s", name);
        end else $display("[ OK ] %s", name);
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

    task automatic wait_for_state(input logic [5:0] state_val, input int max_cycles, output bit ok);
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
        int banks, bank, row;
        banks = dut.u_unified_buffer.BANKS;
        bank = addr % banks;
        row = addr / banks;
        return dut.u_unified_buffer.bank_mem[bank][row];
    endfunction

    initial begin
        bit reached_header;
        bit reached_halt;
        int i;
        logic [15:0] got;

        $display("=================================================");
        $display("A5 BUF_FILL smoke  BUFFER_SIZE=%0d", TB_BUFFER_SIZE);
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        CHECK("BUF_FILL_EN==1", dut.BUF_FILL_EN == 1);
        CHECK("WEIGHT_REGION_WORDS==14144", dut.WEIGHT_REGION_WORDS == 14144);
        CHECK("BUFFER_SIZE holds FC1 payload", TB_BUFFER_SIZE >= 14144);

        // A5 fill 8 words at dest=100
        push_rx_byte(MAGIC_BUF_FILL);
        push_word(TB_DEST[15:0]);
        push_word(TB_COUNT[15:0]);
        for (i = 0; i < TB_COUNT; i++)
            push_word(16'hB000 + i[15:0]);

        wait_for_state(dut.UPLOAD_HEADER_STATE, 200000, reached_header);
        CHECK("Returned to UPLOAD_HEADER after A5", reached_header);
        wait_cycles(4);

        for (i = 0; i < TB_COUNT; i++) begin
            got = peek_buffer_word(TB_DEST + i);
            CHECK($sformatf("buffer[%0d]==0x%04x", TB_DEST + i, 16'hB000 + i[15:0]),
                  got === (16'hB000 + i[15:0]));
        end

        // Legacy A1 HALT still works after A5
        push_rx_byte(MAGIC_UPLOAD);
        push_rx_byte(8'd1);
        push_rx_byte(8'd0);
        push_word(W_HALT);
        wait_for_state(dut.WAIT_START_STATE, 100000, reached_header);
        push_rx_byte(MAGIC_START);
        wait_for_state(dut.HALT_STATE, 100000, reached_halt);
        CHECK("HALT after A1 program post-A5", reached_halt);

        if (errors == 0)
            $display("BUF_FILL_SMOKE_PASS tests=%0d", tests);
        else
            $display("BUF_FILL_SMOKE_FAIL errors=%0d/%0d", errors, tests);
        $finish;
    end
endmodule
