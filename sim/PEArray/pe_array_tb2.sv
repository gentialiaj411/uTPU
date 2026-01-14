
`timescale 1ns/1ps

module pe_array_tb2;

  localparam int ARRAY_SIZE = 2;
  localparam int DW = 4;
  localparam int AW = 16;

  logic clk, rst, compute, load_en;

  logic signed [DW-1:0] ins        [ARRAY_SIZE-1:0];
  logic signed [DW-1:0] weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0];
  logic signed [AW-1:0] results    [ARRAY_SIZE-1:0];

  pe_array #(
    .ARRAY_SIZE(ARRAY_SIZE),
    .COMPUTE_DATA_WIDTH(DW),
    .ACCUMULATOR_DATA_WIDTH(AW)
  ) dut (
    .clk(clk), .rst(rst), .compute(compute), .load_en(load_en),
    .datas_in(ins), .weights_in(weights_in), .results(results)
  );

  // clock
  initial clk = 0;
  always #5 clk = ~clk;

  task automatic clear_ins();
    for (int i = 0; i < ARRAY_SIZE; i++) ins[i] = '0;
  endtask

  task automatic clear_weights_in();
    for (int j = 0; j < ARRAY_SIZE; j++) weights_in[j] = '0;
  endtask

  // Pretty printer: shows the internal mesh at a clock edge
  task automatic dump_state(string tag);
    $display("\n[%0t] %s  rst=%0b load_en=%0b compute=%0b",
             $time, tag, rst, load_en, compute);

    // show inputs
    $write("  ins:        ");
    for (int i=0; i<ARRAY_SIZE; i++) $write("%0d ", ins[i]);
    $write("\n  weights_in: ");
    for (int j=0; j<ARRAY_SIZE*ARRAY_SIZE; j++) $write("%0d ", weights_in[j]);
    $write("\n");



    // Probe PE-local stored weights (the internal reg inside each PE)
    $display("  weights loaded (PE local regs):");
    for (int r = 0; r < ARRAY_SIZE; r++) begin
	$write("    row[%0d]: ", r);
	for (int c = 0; c < ARRAY_SIZE; c++) begin
	    $write("%0d ", dut);
	end
    $write("\n");
    end



    // show activations bus (activations is [ARRAY_SIZE-1:0][ARRAY_SIZE:0])
    $display("  activations bus (cols 0..%0d):", ARRAY_SIZE);
    for (int r=0; r<ARRAY_SIZE; r++) begin
      $write("    a_row[%0d]: ", r);
      for (int c=0; c<=ARRAY_SIZE; c++) $write("%0d ", dut.activations[r][c]);
      $write("\n");
    end

    // show accumulators grid
    $display("  accumulators:");
    for (int r=0; r<ARRAY_SIZE; r++) begin
      $write("    acc[%0d]: ", r);
      for (int c=0; c<ARRAY_SIZE; c++) $write("%0d ", dut.accumulators[r][c]);
      $write("\n");
    end

    // show results
    $write("  results:    ");
    for (int c=0; c<ARRAY_SIZE; c++) $write("%0d ", results[c]);
    $write("\n");
  endtask

  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, pe_array_tb2);
    // init
    rst = 1;
    compute = 0;
    load_en = 0;
    clear_ins();
    clear_weights_in();

    repeat (2) @(posedge clk);
    dump_state("after init (still in reset)");
    rst = 0;

    // -------------------------
    // LOAD WEIGHTS (2x2 example)
    // Want B = [[5,6],[7,8]]
    // Stream row0 weights then row1 weights down the columns.
    // -------------------------
    load_en = 1;

    // cycle L0: inject top row (5,6)
    weights_in[0] = 5;
    weights_in[1] = 6;
    weights_in[2] = 7;
    weights_in[3] = 1;
    @(posedge clk); dump_state("after weight injected weights");

    // extra cycle to settle
    @(posedge clk); dump_state("after weight settle");

    load_en = 0;
    clear_weights_in();

    // -------------------------
    // COMPUTE
    // A = [[1,2],[3,4]]
    // Inject two cycles of activations (no staggering yet; just observe motion)
    // -------------------------
    compute = 1;

    ins[0] = 1; ins[1] = 3;
    @(posedge clk); dump_state("after inject k=0");

    ins[0] = 2; ins[1] = 4;
    @(posedge clk); dump_state("after inject k=1");

    clear_ins();
    repeat (4) begin
      @(posedge clk);
      dump_state("drain");
    end

    compute = 0;
    dump_state("done");
    $finish;
  end

endmodule
