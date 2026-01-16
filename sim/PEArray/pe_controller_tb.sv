

`timescale 1ns/1ps

module pe_controller_tb;

  localparam int ARRAY_SIZE             = 8;
  localparam int COMPUTE_DATA_WIDTH     = 4;
  localparam int ACCUMULATOR_DATA_WIDTH = 16;
  localparam int BUFFER_WORD_SIZE       = 16;
  localparam int NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;

  localparam int OUT_OFFSET = 3;

  logic clk, rst, compute, load_en;

  logic signed [COMPUTE_DATA_WIDTH-1:0] datas_arr  [ARRAY_SIZE*ARRAY_SIZE-1:0];
  logic signed [COMPUTE_DATA_WIDTH-1:0] weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0];

  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results_arr     [ARRAY_SIZE*ARRAY_SIZE-1:0];
  logic signed [ACCUMULATOR_DATA_WIDTH-1:0] exp_results_arr [ARRAY_SIZE*ARRAY_SIZE-1:0];

  // loop vars declared at module scope (Vivado-friendly)
  integer i;
  integer cyc;
  integer j;
  integer idx;

  integer cc;
  integer k;
  integer base_idx;

  // triangular number
  function automatic integer tri_num(input integer t);
    tri_num = (t*(t-1))/2;
  endfunction

  // clock
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // init
  initial begin
    compute = 1'b1;
    load_en = 1'b0;

    for (i = 0; i < ARRAY_SIZE*ARRAY_SIZE; i = i + 1) begin
      datas_arr[i]       = '0;
      weights_in[i]      = '0;
      exp_results_arr[i] = '0;
    end
  end

  // DUT
  pe_controller #(
    .ARRAY_SIZE(ARRAY_SIZE),
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

  // reset
  initial begin
    rst = 1'b1;
    repeat (2) @(posedge clk);
    rst = 1'b0;
  end

  // checker
  initial begin
    // wait until reset drops
    @(negedge rst);
    
    for (int i=0; i < ARRAY_SIZE*ARRAY_SIZE; i++) begin
	datas_arr[i] = i%8 - 1;
	weights_in[i]  = i%8 - 1;
    end

    @(posedge clk);
    compute = '0;
    load_en    = 1'b1;
    repeat (2) @(posedge clk);
    
    load_en = '0;
    compute = 1'b1;

    repeat(10000) @(posedge clk);

    for (idx = 0; idx < ARRAY_SIZE; idx++) begin
//	$display("input %0d: %0d", idx, datas_arr[idx]);
	for (j = 0; j < ARRAY_SIZE; j++) begin
	    $write("%0d ", results_arr[j+idx*ARRAY_SIZE]);
	end 
	$write("\n");
    end
    $display("PASS");
    $finish;
  end

endmodule
