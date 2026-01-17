
`timescale 1ns/1ps

module system_tb;

  logic clk = 0;
  logic rst = 1;
  logic start = 0;
  logic rx = 1'b1;
  wire  tx;

  localparam time CLK_PERIOD = 10ns;
  always #(CLK_PERIOD/2) clk = ~clk;

  top dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .rx(rx),
    .tx(tx)
  );

  initial begin
    // reset
    repeat (5) @(posedge clk);
    rst <= 0;

    // pulse start
    @(posedge clk);
    start <= 1;
    @(posedge clk);
    start <= 0;

    // run for a bit
    repeat (2000) @(posedge clk);

    $display("system_tb finished (smoke). tx=%0b", tx);
    $finish;
  end

endmodule
