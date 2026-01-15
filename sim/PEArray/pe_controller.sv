`timescale 1ns/1ps


`timescale 1ns/1ps

module tb_pe_controller;

  // -------------------------
  // Params (match DUT defaults)
  // -------------------------
  localparam int ARRAY_SIZE            = 8;
  localparam int ARRAY_SIZE_WIDTH      = $clog2(ARRAY_SIZE);
  localparam int COMPUTE_DATA_WIDTH    = 4;
  localparam int ACCUMULATOR_DATA_WIDTH= 16;
  localparam int BUFFER_WORD_SIZE      = 16;
  localparam int NUM_COMPUTE_LANES     = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;

  // -------------------------
  // DUT signals
  // -------------------------
  logic clk, rst, compute, load_en;

  logic signed [COMPUTE_DATA_WIDTH-1:0] datas_arr  [ARRAY_SIZE*ARRAY_SIZE-1:0];
  logic signed [COMPUTE_DATA_WIDTH-1:0] weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0];
  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results_arr [ARRAY_SIZE*ARRAY_SIZE-1:0];

  // -------------------------
  // Instantiate DUT
  // -------------------------
  pe_controller #(
    .ARRAY_SIZE(ARRAY_SIZE),
    .ARRAY_SIZE_WIDTH(ARRAY_SIZE_WIDTH),
    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
    .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
    .BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
    .NUM_COMPUTE_LANES(NUM_COMPUTE_LANES)
  ) dut (
    .clk(clk),
    .rst(rst),
    .compute(compute),
    .load_en(load_en),
    .datas_arr(datas_arr),
    .weights_in(weights_in),
    .results_arr(results_arr)
  );

  // -------------------------
  // Clock
  // -------------------------
  initial clk = 0;
  always #5 clk = ~clk;

  // -------------------------
  // Helpers
  // -------------------------
  function automatic logic signed [COMPUTE_DATA_WIDTH-1:0]
  ref_datas_in(input int lane, input int t);
    int k;
    begin
      // valid when lane <= t < ARRAY_SIZE + lane
      if (t >= lane && t < (ARRAY_SIZE + lane)) begin
        k = t - lane; // "row"/time index
        // datas_arr is flattened row-major: datas_arr[ARRAY_SIZE*k + lane]
        ref_datas_in = datas_arr[ARRAY_SIZE*k + lane];
      end else begin
        ref_datas_in = '0;
      end
    end
  endfunction

  // -------------------------
  // Stimulus
  // -------------------------
  int t;

  initial begin
    // init
    rst     = 1;
    compute = 0;
    load_en = 0;

    // Fill datas_arr with something easy to recognize:
    // value = row*10 + col (fits in 4-bit for small values; we mod 16)
    for (int r = 0; r < ARRAY_SIZE; r++) begin
      for (int c = 0; c < ARRAY_SIZE; c++) begin
        datas_arr[ARRAY_SIZE*r + c] = logic'( (r*10 + c) % 16 );
      end
    end

    // weights don't matter for testing the staggered input generation
    for (int i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i++) begin
      weights_in[i] = '0;
    end

    // release reset after a couple cycles
    repeat (2) @(posedge clk);
    rst <= 0;

    // Let it run for a full skew window (0..(2N-2))
    // Note: your controller's internal wrap length is (2N-1) cycles.
    for (t = 0; t < (ARRAY_SIZE*2 - 1); t++) begin
      @(posedge clk);
      // Give nonblocking assignments a delta cycle to settle
      #1;

      $display("t=%0d", t);
      for (int lane = 0; lane < ARRAY_SIZE; lane++) begin
        // Access DUT's internal datas_in for checking
        // This requires datas_in NOT declared "automatic" and being visible via hierarchy (it is).
        logic signed [COMPUTE_DATA_WIDTH-1:0] got, exp;
        got = dut.datas_in[lane];
        exp = ref_datas_in(lane, t);

        $display("  lane[%0d] got=%0d exp=%0d", lane, got, exp);

        if (got !== exp) begin
          $error("Mismatch at t=%0d lane=%0d: got=%0d exp=%0d",
                 t, lane, got, exp);
        end
      end
      $display("");
    end

    $display("PASS: staggered datas_in matches reference.");
    $finish;
  end

endmodule
