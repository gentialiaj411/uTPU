
`timescale 1ns/1ps

module system_tb;

  logic clk = 0;
  logic rst = 1;
  logic start = 0;
  logic rx = 1'b1;
  wire  tx;

  localparam time CLK_PERIOD = 10ns;
  localparam int  UART_BITS = 8;

  always #(CLK_PERIOD/2) clk = ~clk;

  top dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .rx(rx),
    .tx(tx)
  );

  // -------------------------
  // UART BFM (simplified)
  // -------------------------
  task automatic wait_uart_tick();
    @(posedge dut.u_uart.u_clk_divider.uart_clk);
  endtask

  task automatic uart_send_byte(input logic [UART_BITS-1:0] b);
    int i;
    // start (receiver only looks for rx low to begin)
    rx <= 1'b0;
    wait_uart_tick();
    for (i = 0; i < UART_BITS; i++) begin
      rx <= b[i];
      wait_uart_tick();
    end
    // idle
    rx <= 1'b1;
    wait_uart_tick();
  endtask

  task automatic uart_send_word16(input logic [15:0] w);
    uart_send_byte(w[7:0]);   // LSB first
    uart_send_byte(w[15:8]);  // MSB second
  endtask

  // -------------------------
  // ISA helpers
  // -------------------------
  localparam int OPCODE_WIDTH = 3;
  localparam int ADDRESS_SIZE = 9;

  function automatic logic [15:0] enc_run(
    input logic compute_en,
    input logic quant_en,
    input logic relu_en,
    input logic [ADDRESS_SIZE-1:0] addr
  );
    enc_run = {addr, 1'b0, relu_en, quant_en, compute_en, 3'd2}; // RUN_OP = 2
  endfunction

  function automatic logic [15:0] enc_fetch(
    input logic bot,
    input logic [ADDRESS_SIZE-1:0] addr
  );
    enc_fetch = {addr, 3'b000, bot, 3'd1}; // FETCH_OP = 1
  endfunction

  function automatic logic [15:0] enc_load(
    input logic load_en,
    input logic [ADDRESS_SIZE-1:0] addr
  );
    enc_load = {addr, 3'b000, load_en, 3'd3}; // LOAD_OP = 3
  endfunction

  function automatic logic [15:0] enc_store(
    input logic addr_indicator
  );
    enc_store = {11'b0, addr_indicator, 1'b0, 3'd0}; // STORE_OP = 0
  endfunction

  function automatic logic [15:0] enc_nop();
    enc_nop = {13'b0, 3'd5}; // NOP = 5
  endfunction

  // -------------------------
  // Utilities
  // -------------------------
  task automatic wait_cycles(input int n);
    repeat (n) @(posedge clk);
  endtask

  task automatic dump_status(input string why);
    $display("[%0t] %s state=%0d fetch_mode=%0d instr_half=%0b rx_empty=%0b rx_valid=%0b rx_we=%0b rx_re=%0b",
             $time, why, dut.current_state, dut.fetch_mode, dut.instruction_half,
             dut.rx_empty, dut.rx_valid, dut.rx_we, dut.rx_re);
  endtask

  task automatic wait_for_fetch_ready(input int unsigned max_cycles);
    int i;
    int j;
    for (i = 0; i < max_cycles; i++) begin
      if (dut.current_state == dut.FETCH_FIFO_STATE && dut.instruction_half == 1'b0) begin
        return;
      end
      if (dut.current_state == dut.FETCH_FIFO_STATE && dut.instruction_half == 1'b1) begin
        // complete the half-word with a padding byte to get back to a clean fetch state
        uart_send_byte(8'h00);
        for (j = 0; j < max_cycles; j++) begin
          if (dut.current_state == dut.FETCH_FIFO_STATE && dut.instruction_half == 1'b0) begin
            return;
          end
          @(posedge clk);
        end
        dump_status("fetch_ready padding timeout");
        $fatal(1, "Timeout waiting for FETCH_FIFO_STATE/instruction_half=0 after padding byte");
      end
      @(posedge clk);
    end
    dump_status("fetch_ready timeout");
    $fatal(1, "Timeout waiting for FETCH_FIFO_STATE/instruction_half=0");
  endtask

  task automatic wait_for_buffer_done(input int unsigned max_cycles);
    int i;
    for (i = 0; i < max_cycles; i++) begin
      if (dut.buffer_done) begin
        return;
      end
      @(posedge clk);
    end
    $fatal(1, "Timeout waiting for buffer_done");
  endtask

  task automatic wait_for_compute_done(input int unsigned max_cycles);
    int i;
    for (i = 0; i < max_cycles; i++) begin
      if (dut.compute_done) begin
        return;
      end
      @(posedge clk);
    end
    $fatal(1, "Timeout waiting for compute_done");
  endtask

  function automatic logic [15:0] pack_nibbles(
    input logic [3:0] n0,
    input logic [3:0] n1,
    input logic [3:0] n2,
    input logic [3:0] n3
  );
    pack_nibbles = {n3, n2, n1, n0};
  endfunction

  localparam int LANES = 64;
  task automatic check_array_no_x_4(input logic [3:0] arr [LANES-1:0], input string label);
    int i;
    for (i = 0; i < LANES; i++) begin
      if (^arr[i] === 1'bx) begin
        $fatal(1, "X detected in %s at lane %0d", label, i);
      end
    end
  endtask

  // -------------------------
  // Unit tests for primitives
  // -------------------------
  logic signed [15:0] q_in;
  logic signed [3:0]  q_out;
  quantizer #(.ACCUMULATOR_DATA_WIDTH(16), .COMPUTE_DATA_WIDTH(4)) u_q (
    .in(q_in),
    .result(q_out)
  );

  logic signed [3:0] relu_in, relu_out;
  leaky_relu #(.ALPHA(2), .COMPUTE_DATA_WIDTH(4)) u_relu (
    .in(relu_in),
    .result(relu_out)
  );

  task automatic test_quantizer();
    q_in = 16'sh7FFF;
    #1;
    if (q_out !== 4'sh7) $fatal(1, "quantizer max failed: %0h", q_out);
    q_in = -16'sh8000;
    #1;
    if (q_out !== -4'sh8) $fatal(1, "quantizer min failed: %0h", q_out);
    q_in = 16'sh1000;
    #1;
    if (q_out !== 4'sh1) $fatal(1, "quantizer shift failed: %0h", q_out);
  endtask

  task automatic test_leaky_relu();
    relu_in = 4'sh3;
    #1;
    if (relu_out !== 4'sh3) $fatal(1, "relu positive failed: %0h", relu_out);
    relu_in = -4'sh4;
    #1;
    if (relu_out !== -4'sh1) $fatal(1, "relu negative failed: %0h", relu_out);
  endtask

  // -------------------------
  // System tests (UART-driven)
  // -------------------------
  task automatic test_decode_run();
    logic [15:0] instr;
    wait_for_fetch_ready(2000);
    instr = enc_run(1'b1, 1'b1, 1'b0, 9'h012);
    uart_send_word16(instr);
    wait_cycles(50);
    if (dut.compute_en !== 1'b1) $fatal(1, "RUN decode compute_en failed");
    if (dut.quantizer_en !== 1'b1) $fatal(1, "RUN decode quantizer_en failed");
    if (dut.relu_en !== 1'b0) $fatal(1, "RUN decode relu_en failed");
    if (dut.address !== 9'h012) $fatal(1, "RUN decode address failed: %0h", dut.address);
  endtask

  task automatic test_decode_fetch();
    logic [15:0] instr;
    wait_for_fetch_ready(2000);
    instr = enc_fetch(1'b1, 9'h034);
    uart_send_word16(instr);
    wait_cycles(50);
    if (dut.bot_mem !== 1'b1) $fatal(1, "FETCH decode bot_mem failed");
    if (dut.address !== 9'h034) $fatal(1, "FETCH decode address failed: %0h", dut.address);
  endtask

  task automatic test_decode_load();
    logic [15:0] instr;
    wait_for_fetch_ready(2000);
    instr = enc_load(1'b1, 9'h055);
    uart_send_word16(instr);
    wait_cycles(50);
    if (dut.compute_load_en !== 1'b1) $fatal(1, "LOAD decode compute_load_en failed");
    if (dut.address !== 9'h055) $fatal(1, "LOAD decode address failed: %0h", dut.address);
  endtask

  task automatic test_fetch_buffer_read();
    logic [15:0] word;
    wait_for_fetch_ready(2000);
    word = 16'hA55A;
    dut.u_unified_buffer.mem[9'h000] = word;
    uart_send_word16(enc_fetch(1'b0, 9'h000));
    wait_for_buffer_done(2000);
    if (dut.mem_to_tx_fifo !== word[7:0]) begin
      $fatal(1, "FETCH buffer low byte failed: got %0h exp %0h", dut.mem_to_tx_fifo, word[7:0]);
    end
  endtask

  task automatic test_load_buffer_to_compute();
    int i;
    logic [3:0] expected [LANES-1:0];
    logic [3:0] n0, n1, n2, n3;
    wait_for_fetch_ready(2000);
    for (i = 0; i < 16; i++) begin
      n0 = (i*4 + 0) & 4'hF;
      n1 = (i*4 + 1) & 4'hF;
      n2 = (i*4 + 2) & 4'hF;
      n3 = (i*4 + 3) & 4'hF;
      expected[i*4 + 0] = n0;
      expected[i*4 + 1] = n1;
      expected[i*4 + 2] = n2;
      expected[i*4 + 3] = n3;
      dut.u_unified_buffer.mem[9'h010 + i] = pack_nibbles(n0, n1, n2, n3);
    end
    uart_send_word16(enc_load(1'b1, 9'h010));
    wait_for_buffer_done(4000);
    for (i = 0; i < LANES; i++) begin
      if (dut.mem_to_compute[i] !== expected[i]) begin
        $fatal(1, "LOAD mem_to_compute mismatch at %0d: got %0h exp %0h", i, dut.mem_to_compute[i], expected[i]);
      end
    end
    check_array_no_x_4(dut.mem_to_compute, "mem_to_compute");
  endtask

  task automatic test_nop();
    wait_for_fetch_ready(2000);
    uart_send_word16(enc_nop());
    wait_cycles(20);
  endtask

  initial begin
    // reset
    repeat (5) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    start <= 1;
    @(posedge clk);
    start <= 0;

    test_quantizer();
    test_leaky_relu();
    test_nop();
    test_decode_run();
    test_decode_fetch();
    test_decode_load();
    test_fetch_buffer_read();
    test_load_buffer_to_compute();

    $display("system_tb finished (full). tx=%0b", tx);
    $finish;
  end

endmodule
