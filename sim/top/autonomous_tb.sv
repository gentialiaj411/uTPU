`timescale 1ns/1ps

// Testbench for autonomous-inference FSM (upload → start → execute).
//
// Bytes are injected directly into the RX FIFO by forcing rx_data_buf and
// rx_we_d — bypassing UART bit-time serialization for fast simulation.
//
// TX output is checked by observing dut.tx_pending_data when dut.tx_pending
// goes high (the cycle FETCH_BUFFER_STATE finishes), avoiding any need to
// drain the TX FIFO or model the UART transmitter.
//
// Tests:
//   1. Reset → UPLOAD_HEADER_STATE
//   2. Upload a HALT program → FSM reaches HALT_STATE
//   3. Re-arm (0xA3) → back to UPLOAD_HEADER_STATE
//   4. Four NOPs then HALT → correct execution trace
//   5. BRAM holds correct words after upload
//   6. STORE (immediate) → FETCH_BUFFER → tx_pending_data correct
//   7. LOAD_STATE is reached from LOAD instruction

module autonomous_tb;

    // -----------------------------------------------------------------------
    // Clock / reset
    // -----------------------------------------------------------------------
    logic clk = 0;
    logic rst = 0;   // 0 = in reset (active-low external, rst_int = ~rst = 1)
    localparam time CLK_PERIOD = 10ns;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // DUT — tiny 2×2 array, small buffer/BRAM so synthesis fits quickly
    // -----------------------------------------------------------------------
    localparam int TB_ARRAY_SIZE  = 2;
    localparam int TB_BUFFER_SIZE = 32;   // ADDRESS_SIZE = 5
    localparam int TB_FIFO_WIDTH  = 64;
    localparam int TB_PROG_DEPTH  = 64;   // PC_WIDTH = 6

    logic rx = 1'b1;
    wire  tx;
    wire  led_rst;

    top #(
        .ARRAY_SIZE  (TB_ARRAY_SIZE),
        .BUFFER_SIZE (TB_BUFFER_SIZE),
        .FIFO_WIDTH  (TB_FIFO_WIDTH),
        .PROG_DEPTH  (TB_PROG_DEPTH)
    ) dut (
        .clk(clk), .rst(rst),
        .rx(rx), .tx(tx),
        .led_rst(led_rst)
    );

    // -----------------------------------------------------------------------
    // Scoreboard
    // -----------------------------------------------------------------------
    int tests  = 0;
    int errors = 0;

    task automatic CHECK(input string name, input bit cond);
        tests++;
        if (!cond) begin
            errors++;
            $display("[FAIL] %s  (t=%0t)", name, $time);
        end else begin
            $display("[ OK ] %s", name);
        end
    endtask

    task automatic FINISH();
        $display("=================================================");
        $display("DONE: tests=%0d  errors=%0d", tests, errors);
        $display("=================================================");
        if (errors != 0) $fatal(1, "TB finished with errors");
        $finish;
    endtask

    // -----------------------------------------------------------------------
    // RX injection: push one byte directly into fifo_in by forcing the
    // registered write-path signals (rx_data_buf, rx_we_d) for one clock.
    // This bypasses UART bit-time serialization.
    // -----------------------------------------------------------------------
    task automatic push_rx_byte(input logic [7:0] b);
        // Drive just after a posedge so the forced values are stable before
        // the next posedge captures them into fifo_rx.
        @(posedge clk); #1;
        force dut.rx_data_buf = b;
        force dut.rx_we_d     = 1'b1;
        @(posedge clk); #1;   // fifo_rx writes on this edge (write_ok = we && !full)
        release dut.rx_data_buf;
        release dut.rx_we_d;
        @(posedge clk);        // let FSM process the new byte via rx_re/rx_rvalid
        @(posedge clk);
    endtask

    task automatic push_rx_word16(input logic [15:0] w);
        push_rx_byte(w[7:0]);
        push_rx_byte(w[15:8]);
    endtask

    // -----------------------------------------------------------------------
    // Wait helpers
    // -----------------------------------------------------------------------
    task automatic wait_state(input dut.state_e s, input int max_cycles = 1000);
        int i;
        for (i = 0; i < max_cycles; i++) begin
            if (dut.current_state == s) return;
            @(posedge clk);
        end
        $fatal(1, "Timeout waiting for state %s (stuck in %s, t=%0t)",
               s.name(), dut.current_state.name(), $time);
    endtask

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // ISA helpers — correct 16-bit encoding for TB parameters
    //
    //   ADDRESS_SIZE = $clog2(TB_BUFFER_SIZE) = 5
    //   instruction[15:11] = address (bits BUFFER_WORD_SIZE-1 downto BUFFER_WORD_SIZE-ADDRESS_SIZE)
    //   This matches how DECODE_STATE slices the instruction in top.sv.
    //
    // Bit positions (constant across all ADDRESS_SIZE values):
    //   [2:0]  = opcode
    //   [3]    = flag0 (bot_mem / is_weights / compute_en / top_half)
    //   [4]    = flag1 (address_indicator / quantizer_en)
    //   [5]    = flag2 (relu_en)
    //   [6]    = unused
    //   [15:7] = address (only [15:11] used with 5-bit ADDRESS_SIZE)
    // -----------------------------------------------------------------------
    localparam int ADDR_W = $clog2(TB_BUFFER_SIZE); // 5

    function automatic logic [15:0] enc_nop();
        enc_nop = 16'(3'd5);
    endfunction

    function automatic logic [15:0] enc_halt();
        enc_halt = 16'(3'd4);
    endfunction

    // STORE immediate: opcode=0, bit4=1 (address_indicator=1 → immediate mode)
    function automatic logic [15:0] enc_store_imm();
        enc_store_imm = 16'h0010; // bit4=1, opcode=000
    endfunction

    // FETCH: opcode=1, bit3=bot_mem, addr in [15:11]
    function automatic logic [15:0] enc_fetch(
        input logic [ADDR_W-1:0] addr,
        input logic               bot   // 0 = top byte [15:8], 1 = bottom byte [7:0]
    );
        // {addr[4:0], 7'b0, bot, 3'b001} = 16 bits
        enc_fetch = {addr, 7'b000_0000, bot, 3'd1};
    endfunction

    // LOAD: opcode=3, bit3=is_weights, addr in [15:11]
    function automatic logic [15:0] enc_load(
        input logic [ADDR_W-1:0] addr,
        input logic               is_weights
    );
        enc_load = {addr, 7'b000_0000, is_weights, 3'd3};
    endfunction

    // RUN: opcode=2, bit3=compute_en, bit4=quant_en, bit5=relu_en, addr in [15:11]
    function automatic logic [15:0] enc_run(
        input logic [ADDR_W-1:0] addr,
        input logic               compute_en,
        input logic               quant_en,
        input logic               relu_en
    );
        enc_run = {addr, 5'b0, relu_en, quant_en, compute_en, 3'd2};
    endfunction

    // -----------------------------------------------------------------------
    // Upload protocol helpers
    // -----------------------------------------------------------------------
    task automatic upload_program(input logic [15:0] prog [], input int n_words);
        int i;
        // 0xA3 re-arm (ignored in UPLOAD_HEADER, unblocks HALT_STATE)
        push_rx_byte(8'hA3);
        // 0xA1 magic
        push_rx_byte(8'hA1);
        // 2-byte LE length
        push_rx_byte(8'(n_words));
        push_rx_byte(8'(n_words >> 8));
        // instruction words, little-endian
        for (i = 0; i < n_words; i++) begin
            push_rx_byte(prog[i][7:0]);
            push_rx_byte(prog[i][15:8]);
        end
    endtask

    task automatic send_start();
        push_rx_byte(8'hA2);
    endtask

    // -----------------------------------------------------------------------
    // Test 1 — reset releases into UPLOAD_HEADER_STATE
    // -----------------------------------------------------------------------
    task automatic test_reset_state();
        $display("\n---- test_reset_state ----");
        CHECK("in UPLOAD_HEADER after reset release",
              dut.current_state == dut.UPLOAD_HEADER_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 2 — minimal HALT program: upload → start → HALT_STATE
    // -----------------------------------------------------------------------
    task automatic test_halt_program();
        logic [15:0] prog[1];
        $display("\n---- test_halt_program ----");
        prog[0] = enc_halt();
        upload_program(prog, 1);
        wait_state(dut.WAIT_START_STATE, 200);
        send_start();
        wait_state(dut.HALT_STATE, 200);
        CHECK("FSM reaches HALT_STATE", dut.current_state == dut.HALT_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 3 — re-arm from HALT → UPLOAD_HEADER_STATE
    // -----------------------------------------------------------------------
    task automatic test_rearm();
        $display("\n---- test_rearm ----");
        // While in HALT_STATE, send 0xA3
        push_rx_byte(8'hA3);
        wait_state(dut.UPLOAD_HEADER_STATE, 50);
        CHECK("re-arm returns to UPLOAD_HEADER",
              dut.current_state == dut.UPLOAD_HEADER_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 4 — four NOPs then HALT, verify PC advanced correctly
    // -----------------------------------------------------------------------
    task automatic test_nop_sequence();
        logic [15:0] prog[5];
        $display("\n---- test_nop_sequence ----");
        prog[0] = enc_nop();
        prog[1] = enc_nop();
        prog[2] = enc_nop();
        prog[3] = enc_nop();
        prog[4] = enc_halt();
        upload_program(prog, 5);
        wait_state(dut.WAIT_START_STATE, 200);
        send_start();
        wait_state(dut.HALT_STATE, 400);
        CHECK("HALT after 4 NOPs", dut.current_state == dut.HALT_STATE);
        // PC was incremented once per FETCH_BRAM_STATE; HALT is at word 4 so
        // PC = 5 after the HALT fetch.
        CHECK("PC advanced to 5", dut.pc == 5);
    endtask

    // -----------------------------------------------------------------------
    // Test 5 — BRAM contents match uploaded program
    // -----------------------------------------------------------------------
    task automatic test_bram_contents();
        logic [15:0] prog[4];
        $display("\n---- test_bram_contents ----");
        prog[0] = enc_nop();
        prog[1] = enc_nop();
        prog[2] = enc_nop();
        prog[3] = enc_halt();
        upload_program(prog, 4);
        wait_state(dut.WAIT_START_STATE, 300);
        // Check BRAM contents via hierarchical reference (before execution)
        CHECK("BRAM[0]=NOP",  dut.u_instr_bram.mem[0] === enc_nop());
        CHECK("BRAM[1]=NOP",  dut.u_instr_bram.mem[1] === enc_nop());
        CHECK("BRAM[2]=NOP",  dut.u_instr_bram.mem[2] === enc_nop());
        CHECK("BRAM[3]=HALT", dut.u_instr_bram.mem[3] === enc_halt());
        // Run it to HALT
        send_start();
        wait_state(dut.HALT_STATE, 200);
        CHECK("Ran NOP×3+HALT to completion",
              dut.current_state == dut.HALT_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 6 — STORE (immediate) sets store_val and store_dest_addr correctly,
    //          then FETCH_BUFFER fires and tx_pending_data matches the stored value.
    //
    // Program: STORE imm 0xABCD → addr 0  |  FETCH addr 0 bot=0  |  HALT
    // Expected: after FETCH_BUFFER_STATE, tx_pending=1 and
    //           tx_pending_data = low byte of 0xABCD = 0xCD
    // -----------------------------------------------------------------------
    task automatic test_store_fetch();
        logic [15:0] prog[5];
        logic [7:0]  captured;
        int          limit;
        $display("\n---- test_store_fetch ----");
        prog[0] = enc_store_imm();   // STORE opcode, bit4=1
        prog[1] = 16'hABCD;          // value word
        prog[2] = 16'h0000;          // dest addr = 0
        // FETCH addr=0, bot=0 → reads low byte [7:0] of the stored 16-bit word
        prog[3] = enc_fetch(5'h0, 1'b0);
        prog[4] = enc_halt();

        upload_program(prog, 5);
        wait_state(dut.WAIT_START_STATE, 300);
        send_start();

        // Wait for tx_pending to go high (set in FETCH_BUFFER_STATE after buffer_done_d)
        limit = 1000;
        while (~dut.tx_pending && limit > 0) begin
            @(posedge clk);
            limit--;
        end
        CHECK("tx_pending fires after FETCH_BUFFER", dut.tx_pending === 1'b1);
        captured = dut.tx_pending_data;

        // The unified_buffer returns the bottom byte when section=0 (bot_mem=0):
        // stored word 0xABCD → low byte = 0xCD
        CHECK("STORE→FETCH low byte correct", captured === 8'hCD);

        // Let execution finish
        wait_state(dut.HALT_STATE, 200);
        CHECK("STORE→FETCH→HALT completes", dut.current_state == dut.HALT_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 7 — LOAD instruction transitions FSM through LOAD_STATE
    // -----------------------------------------------------------------------
    task automatic test_load_state_reached();
        logic [15:0] prog[2];
        $display("\n---- test_load_state_reached ----");
        // LOAD from addr 0 as weights; then HALT
        prog[0] = enc_load(5'h0, 1'b1);
        prog[1] = enc_halt();

        upload_program(prog, 2);
        wait_state(dut.WAIT_START_STATE, 200);
        send_start();

        // Should pass through LOAD_STATE on the way to HALT
        wait_state(dut.LOAD_STATE, 200);
        CHECK("LOAD_STATE reached", dut.current_state == dut.LOAD_STATE);
        wait_state(dut.HALT_STATE, 400);
        CHECK("HALT after LOAD", dut.current_state == dut.HALT_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Test 8 — STORE word2/word3 multi-word BRAM fetch
    //          Verify STORE_DECIDE_STATE latches correct dest addr and value.
    // -----------------------------------------------------------------------
    task automatic test_store_decide();
        logic [15:0] prog[4];
        $display("\n---- test_store_decide ----");
        // STORE imm 0x1234 → dest addr 5
        prog[0] = enc_store_imm();
        prog[1] = 16'h1234;   // store_val
        prog[2] = 16'h0005;   // dest addr = 5
        prog[3] = enc_halt();

        upload_program(prog, 4);
        wait_state(dut.WAIT_START_STATE, 300);
        send_start();

        // Wait for STORE_DECIDE_STATE to latch word3
        wait_state(dut.STORE_DECIDE_STATE, 200);
        @(posedge clk); // let the sequential block update
        CHECK("store_word2 = 0x1234",    dut.store_word2   === 16'h1234);
        CHECK("store_dest_addr = 5",     dut.store_dest_addr === 5'd5);
        CHECK("address_indicator = 1",   dut.address_indicator === 1'b1);

        wait_state(dut.HALT_STATE, 200);
        CHECK("HALT after STORE", dut.current_state == dut.HALT_STATE);
    endtask

    // -----------------------------------------------------------------------
    // Main
    // -----------------------------------------------------------------------
    initial begin
        $display("=================================================");
        $display("  uTPU Autonomous Inference Testbench");
        $display("=================================================");

        // Reset: rst=0 → rst_int=1 → in reset
        rst <= 0;
        repeat (10) @(posedge clk);
        rst <= 1;   // release reset → rst_int=0
        repeat (5)  @(posedge clk);

        test_reset_state();
        test_halt_program();
        test_rearm();
        test_nop_sequence();
        test_bram_contents();
        test_store_fetch();
        test_load_state_reached();
        test_store_decide();

        FINISH();
    end

endmodule: autonomous_tb
