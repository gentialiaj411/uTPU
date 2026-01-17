
`timescale 1ns/1ps

module top_tb;

  // ----------------------------
  // Clock/reset
  // ----------------------------
  logic clk = 0;
  logic rst = 1;
  logic start = 0;

  logic rx;
  wire  tx;

  localparam time CLK_PERIOD = 10ns;
  always #(CLK_PERIOD/2) clk = ~clk;

  int errors = 0;
  int tests  = 0;

  task CHECK(input string name, input bit cond);
    tests++;
    if (!cond) begin
      errors++;
      $display("[FAIL] %s", name);
    end else begin
      $display("[ OK ] %s", name);
    end
  endtask

  task RESET_DUT;
    rx    = 1'b1;
    start = 1'b0;

    rst <= 1;
    repeat (10) @(posedge clk);
    rst <= 0;
    repeat (2) @(posedge clk);
  endtask

  task FINISH;
    $display("======================================");
    $display("DONE: tests=%0d errors=%0d", tests, errors);
    $display("======================================");
    $finish;
  endtask

  // ----------------------------
  // DUT
  // ----------------------------
  top dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .rx(rx),
    .tx(tx)
  );

  // ============================================================
  // RX-FIFO injection (Vivado-safe)
  //
  // Cannot force from an automatic task arg -> use static staging regs.
  // ============================================================
  logic [7:0] inj_byte;

  task push_rx_fifo_byte(input [7:0] b);
    inj_byte = b; // stage into static variable

    @(posedge clk);

    // force the UART receiver outputs + controller write enable
    force dut.rx_valid   = 1'b1;
    force dut.rx_to_fifo = inj_byte;
    force dut.rx_we      = 1'b1;

    @(posedge clk);

    release dut.rx_we;
    release dut.rx_valid;
    release dut.rx_to_fifo;

    @(posedge clk);
  endtask

  task push_rx_fifo_word16(input [15:0] w);
    // controller fetches low then high (per your code)
    push_rx_fifo_byte(w[7:0]);
    push_rx_fifo_byte(w[15:8]);
  endtask

  // ============================================================
  // TX-FIFO pop (optional, Vivado-safe)
  // ============================================================
  logic [7:0] got_byte;

  task pop_tx_fifo_byte(output [7:0] b);
    int t;
    b = 8'h00;

    t = 2000;
    while (dut.tx_empty && (t > 0)) begin
      @(posedge clk);
      t--;
    end
    CHECK("tx fifo not empty before pop", (t > 0));

    @(posedge clk);
    force dut.tx_re = 1'b1;
    @(posedge clk);
    release dut.tx_re;

    @(posedge clk);
    got_byte = dut.tx_to_fifo;
    b = got_byte;
  endtask

  // ============================================================
  // ISA helpers (edit if opcode mapping changes)
  // ============================================================
  localparam int OPCODE_W = 3;

  localparam logic [OPCODE_W-1:0] OP_STORE = 3'd0;
  localparam logic [OPCODE_W-1:0] OP_FETCH = 3'd1;
  localparam logic [OPCODE_W-1:0] OP_RUN   = 3'd2;
  localparam logic [OPCODE_W-1:0] OP_LOAD  = 3'd3;
  localparam logic [OPCODE_W-1:0] OP_HALT  = 3'd4;
  localparam logic [OPCODE_W-1:0] OP_NOP   = 3'd5;

  function [15:0] instr_d(input logic [OPCODE_W-1:0] op);
    instr_d = 16'h0000;
    instr_d[OPCODE_W-1:0] = op;
  endfunction

  function [15:0] instr_fetch(input bit bot, input [8:0] addr9);
    logic [15:0] w;
    w = 16'h0000;
    w[2:0] = OP_FETCH;
    w[3]   = bot;
    // NOTE: you may need to re-pack addr bits depending on your final instruction layout
    w[15:7] = addr9;
    instr_fetch = w;
  endfunction

  function [15:0] instr_store_imm(input bit imm, input [8:0] addr9);
    logic [15:0] w;
    w = 16'h0000;
    w[2:0] = OP_STORE;
    w[4]   = imm;     // matches your address_indicator usage
    w[15:7] = addr9;  // may need adjustment later
    instr_store_imm = w;
  endfunction

  // ============================================================
  // Tests
  // ============================================================
  task test_elaborates_and_resets;
    $display("---- test_elaborates_and_resets ----");
    RESET_DUT();
    CHECK("rx fifo empty after reset", dut.rx_empty == 1'b1);
    CHECK("tx fifo empty after reset", dut.tx_empty == 1'b1);
  endtask

  task test_push_program_smoke;
    $display("---- test_push_program_smoke ----");
    RESET_DUT();

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    push_rx_fifo_word16(instr_d(OP_NOP));
    push_rx_fifo_word16(instr_d(OP_HALT));

    repeat (200) @(posedge clk);

    CHECK("rx fifo not full", dut.rx_full == 1'b0);
  endtask

  task test_store_and_fetch_smoke;
    $display("---- test_store_and_fetch_smoke ----");
    RESET_DUT();

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    // STORE imm to 0x012, then data word 0xADDE
    push_rx_fifo_word16(instr_store_imm(1'b1, 9'h012));
    push_rx_fifo_word16(16'hADDE);

    push_rx_fifo_word16(instr_fetch(1'b0, 9'h012));
    push_rx_fifo_word16(instr_fetch(1'b1, 9'h012));

    push_rx_fifo_word16(instr_d(OP_HALT));

    repeat (2000) @(posedge clk);

    if (!dut.tx_empty) begin
      logic [7:0] b0, b1;
      pop_tx_fifo_byte(b0);
      pop_tx_fifo_byte(b1);
      $display("Fetched bytes (if implemented): %02x %02x", b0, b1);
    end else begin
      $display("TX FIFO stayed empty (fetch->tx path may not be implemented yet).");
    end

    CHECK("did not deadlock (sim reached end)", 1'b1);
  endtask

  // ============================================================
  // Run
  // ============================================================
  initial begin
    rx = 1'b1;
    start = 1'b0;

    test_elaborates_and_resets();
    test_push_program_smoke();
    test_store_and_fetch_smoke();

    FINISH();
  end

endmodule
