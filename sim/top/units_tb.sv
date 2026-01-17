
`timescale 1ns/1ps

module units_tb;

  // ============================================================
  // Clock / reset
  // ============================================================
  logic clk = 0;
  logic rst = 1;

  localparam time CLK_PERIOD = 10ns;
  always #(CLK_PERIOD/2) clk = ~clk;

  int errors = 0;
  int tests  = 0;

  task automatic CHECK(input string name, input bit cond);
    tests++;
    if (!cond) begin
      errors++;
      $display("[FAIL] %s", name);
    end else begin
      $display("[ OK ] %s", name);
    end
  endtask

  task automatic RESET_DUT(input int cycles = 5);
    rst <= 1;
    repeat (cycles) @(posedge clk);
    rst <= 0;
    @(posedge clk);
  endtask

  task automatic FINISH();
    $display("======================================");
    $display("DONE: tests=%0d errors=%0d", tests, errors);
    $display("======================================");
    $finish;
  endtask

  // ============================================================
  // 1) fifo.sv
  // ============================================================
  localparam int FIFO_WIDTH      = 256;
  localparam int FIFO_DATA_WIDTH = 8;

  logic fifo_we, fifo_re, fifo_empty, fifo_full;
  logic [FIFO_DATA_WIDTH-1:0] fifo_w_data, fifo_r_data;

  fifo #(
    .FIFO_WIDTH(FIFO_WIDTH),
    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
  ) dut_fifo (
    .clk(clk), .rst(rst),
    .we(fifo_we), .re(fifo_re),
    .empty(fifo_empty), .full(fifo_full),
    .w_data(fifo_w_data),
    .r_data(fifo_r_data)
  );

  // static scoreboard (Vivado-safe)
  byte sb_mem [0:FIFO_WIDTH-1];
  int  sb_wptr, sb_rptr, sb_count;

  task automatic test_fifo_basic();
    $display("---- test_fifo_basic ----");
    fifo_we = 0; fifo_re = 0; fifo_w_data = '0;

    RESET_DUT();

    CHECK("fifo empty after reset", fifo_empty == 1);
    CHECK("fifo not full after reset", fifo_full == 0);

    // write 4 bytes
    for (int i = 0; i < 4; i++) begin
      @(posedge clk);
      fifo_we     <= 1;
      fifo_w_data <= i[7:0];
    end
    @(posedge clk);
    fifo_we <= 0;

    CHECK("fifo not empty after writes", fifo_empty == 0);

    // read 4 bytes
    for (int i = 0; i < 4; i++) begin
      @(posedge clk);
      fifo_re <= 1;
      @(posedge clk);
      CHECK($sformatf("fifo read == %0d", i), fifo_r_data == i[7:0]);
    end
    @(posedge clk);
    fifo_re <= 0;

    CHECK("fifo empty after reads", fifo_empty == 1);
  endtask

  task automatic test_fifo_random(input int iters = 2000);
    $display("---- test_fifo_random ----");
    fifo_we = 0; fifo_re = 0; fifo_w_data = '0;

    RESET_DUT();

    sb_wptr  = 0;
    sb_rptr  = 0;
    sb_count = 0;

    for (int t = 0; t < iters; t++) begin
      bit  do_w;
      bit  do_r;
      byte v;
      byte exp;

      do_w = $urandom_range(0,1);
      do_r = $urandom_range(0,1);

      if (fifo_full)  do_w = 0;
      if (fifo_empty) do_r = 0;

      if (sb_count == FIFO_WIDTH) do_w = 0;
      if (sb_count == 0)          do_r = 0;

      // present enables for one cycle
      @(posedge clk);
      fifo_we <= do_w;
      fifo_re <= do_r;

      if (do_w) begin
        v = $urandom();
        fifo_w_data <= v;

        sb_mem[sb_wptr] = v;
        sb_wptr = (sb_wptr + 1) % FIFO_WIDTH;
        sb_count++;
      end

      // drop enables
      @(posedge clk);
      fifo_we <= 0;
      fifo_re <= 0;

      // IMPORTANT: your fifo reads r_data *on the same clock edge* as re.
      // So the value becomes valid right after that edge. We sample here.
      if (do_r) begin
        exp = sb_mem[sb_rptr];
        sb_rptr = (sb_rptr + 1) % FIFO_WIDTH;
        sb_count--;

        CHECK("fifo random read matches", fifo_r_data == exp);
      end

      CHECK("!(empty && full)", !(fifo_empty && fifo_full));
    end
  endtask

  // ============================================================
  // 2) fifo_rx.sv + fifo_tx.sv
  //   NOTE: fifo_rx has a VALID input. Your earlier TB didn’t drive it.
  // ============================================================
  logic rx_we, rx_re, rx_valid, rx_empty, rx_full;
  logic [FIFO_DATA_WIDTH-1:0] rx_w_data, rx_r_data;

  fifo_rx #(
    .FIFO_WIDTH(FIFO_WIDTH),
    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
  ) dut_fifo_rx (
    .clk(clk), .rst(rst),
    .we(rx_we), .re(rx_re), .valid(rx_valid),
    .empty(rx_empty), .full(rx_full),
    .w_data(rx_w_data), .r_data(rx_r_data)
  );

  logic tx_we, tx_re, tx_empty, tx_full, tx_start;
  logic [FIFO_DATA_WIDTH-1:0] tx_w_data, tx_r_data;

  fifo_tx #(
    .FIFO_WIDTH(FIFO_WIDTH),
    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
  ) dut_fifo_tx (
    .clk(clk), .rst(rst),
    .we(tx_we), .re(tx_re),
    .start(tx_start),
    .empty(tx_empty), .full(tx_full),
    .w_data(tx_w_data), .r_data(tx_r_data)
  );

  task automatic test_fifo_rx_tx();
    $display("---- test_fifo_rx_tx ----");

    rx_we = 0; rx_re = 0; rx_valid = 0; rx_w_data = '0;
    tx_we = 0; tx_re = 0; tx_w_data = '0;

    RESET_DUT();

    // fifo_rx write 3 bytes (valid must be 1 for writes to occur)
    for (int i = 0; i < 3; i++) begin
      @(posedge clk);
      rx_valid <= 1;
      rx_we    <= 1;
      rx_w_data <= (8'hA0 + i);
    end
    @(posedge clk);
    rx_we    <= 0;
    rx_valid <= 0;

    CHECK("fifo_rx not empty", rx_empty == 0);

    // fifo_rx read 3 bytes
    for (int i = 0; i < 3; i++) begin
      @(posedge clk);
      rx_re <= 1;
      @(posedge clk);
      rx_re <= 0;
      CHECK("fifo_rx pop matches", rx_r_data == (8'hA0 + i));
    end

    CHECK("fifo_rx empty", rx_empty == 1);

    // fifo_tx basic push/pop
    CHECK("fifo_tx empty after reset", tx_empty == 1);

    @(posedge clk);
    tx_we <= 1;
    tx_w_data <= 8'h55;
    @(posedge clk);
    tx_we <= 0;

    CHECK("fifo_tx not empty after write", tx_empty == 0);

    @(posedge clk);
    tx_re <= 1;
    @(posedge clk);
    tx_re <= 0;

    CHECK("fifo_tx read matches", tx_r_data == 8'h55);
    $display("fifo_tx tx_start=%0b (informational)", tx_start);
  endtask

  // ============================================================
  // Run unit tests
  // ============================================================
  initial begin
    fifo_we=0; fifo_re=0; fifo_w_data='0;
    rx_we=0; rx_re=0; rx_valid=0; rx_w_data='0;
    tx_we=0; tx_re=0; tx_w_data='0;

    test_fifo_basic();
    test_fifo_random(2000);
    test_fifo_rx_tx();

    FINISH();
  end

endmodule
