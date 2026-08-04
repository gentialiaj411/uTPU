`timescale 1ns/1ps

// Parameterized PROG_DEPTH smoke: upload + execute + HALT with a STORE that
// leaves a known word in the unified buffer. Catches upload-path width bugs
// that only showed up as synth pruning (65536) or hard OOR (131072).
//
// Build with: iverilog ... -DTB_PROG_DEPTH=<depth>
module tb_prog_depth_smoke;
`ifndef TB_PROG_DEPTH
    `define TB_PROG_DEPTH 8192
`endif
    localparam int TB_PROG_DEPTH = `TB_PROG_DEPTH;

    logic clk = 0;
    logic rst = 0;
    logic rx = 1'b1;
    wire tx;
    wire led_rst;

    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Match shipping Artix datapath widths used in prog_depth_sweep.
    localparam int TB_ARRAY_SIZE = 8;
    localparam int TB_BUFFER_SIZE = 4096;
    localparam int TB_COMPUTE_DW = 8;
    localparam int TB_ACC_DW = 32;
    localparam int TB_FIFO_WIDTH = 256;
    localparam int TB_FIFO_PTR_W = $clog2(TB_FIFO_WIDTH);

    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1;
    localparam logic [7:0] MAGIC_START  = 8'hA2;

    // STORE-immediate (EXT_ADDR_EN=1 shipping layout still uses 3-word STORE):
    // word1 = OPCODE_STORE | (1<<4) = 0x0010
    // word2 = payload 0xA5A5
    // word3 = dest address 0
    // HALT = 0x0004
    localparam logic [15:0] W_STORE_HDR = 16'h0010;
    localparam logic [15:0] W_STORE_VAL = 16'hA5A5;
    localparam logic [15:0] W_STORE_ADDR = 16'h0000;
    localparam logic [15:0] W_HALT = 16'h0004;
    localparam int PROG_WORDS = 4;

    int tests = 0;
    int errors = 0;

    top #(
        .ARRAY_SIZE(TB_ARRAY_SIZE),
        .BUFFER_SIZE(TB_BUFFER_SIZE),
        .COMPUTE_DATA_WIDTH(TB_COMPUTE_DW),
        .ACCUMULATOR_DATA_WIDTH(TB_ACC_DW),
        .EXT_ADDR_EN(1),
        .PROG_DEPTH(TB_PROG_DEPTH),
        .FIFO_WIDTH(TB_FIFO_WIDTH),
        .MAX_BATCH_COUNT(4),
        .QUANTIZER_PIPE_DEPTH(1)
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

`ifdef DUMP_VCD
        $dumpfile("build/sim_iverilog/tb_prog_depth_smoke.vcd");
        $dumpvars(0, tb_prog_depth_smoke);
`endif

        $display("=================================================");
        $display("PROG_DEPTH smoke TB  depth=%0d", TB_PROG_DEPTH);
        $display("=================================================");

        rst <= 0;
        wait_cycles(10);
        rst <= 1;
        wait_cycles(20);

        // Sanity: UPLOAD_LEN_MAX must admit a 4-word program at every supported depth.
        CHECK("UPLOAD_LEN_MAX covers PROG_WORDS", dut.UPLOAD_LEN_MAX >= PROG_WORDS);
        CHECK("PC_WIDTH matches clog2(PROG_DEPTH)",
              dut.PC_WIDTH == $clog2(TB_PROG_DEPTH));

        push_rx_byte(MAGIC_UPLOAD);
        push_rx_byte(PROG_WORDS[7:0]);
        push_rx_byte(PROG_WORDS[15:8]);
        push_word(W_STORE_HDR);
        push_word(W_STORE_VAL);
        push_word(W_STORE_ADDR);
        push_word(W_HALT);

        wait_for_state(dut.WAIT_START_STATE, 100000, reached_wait_start);
        CHECK("Reached WAIT_START after upload", reached_wait_start);
        // Last upload write uses registered bram_wr_en → commits one cycle after
        // the UPLOAD_BODY beat that transitions to WAIT_START.
        wait_cycles(2);

        // Instr BRAM must hold the uploaded image (proves wr_addr/PC_WIDTH path).
        CHECK("instr_bram[0]=STORE_HDR", dut.u_instr_bram.mem[0] === W_STORE_HDR);
        CHECK("instr_bram[1]=STORE_VAL", dut.u_instr_bram.mem[1] === W_STORE_VAL);
        CHECK("instr_bram[2]=STORE_ADDR", dut.u_instr_bram.mem[2] === W_STORE_ADDR);
        CHECK("instr_bram[3]=HALT", dut.u_instr_bram.mem[3] === W_HALT);

        push_rx_byte(MAGIC_START);
        wait_for_state(dut.HALT_STATE, 200000, reached_halt);
        CHECK("Reached HALT after start", reached_halt);

        got = peek_buffer_word(0);
        CHECK("buffer[0]==0xA5A5 after STORE", got === W_STORE_VAL);

        if (errors == 0)
            $display("PROG_DEPTH_SMOKE_PASS depth=%0d tests=%0d", TB_PROG_DEPTH, tests);
        else
            $display("PROG_DEPTH_SMOKE_FAIL depth=%0d errors=%0d/%0d", TB_PROG_DEPTH, errors, tests);
        $finish;
    end
endmodule
