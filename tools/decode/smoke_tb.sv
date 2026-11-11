module smoke_tb;
  logic [7:0] a=8'd19, b=8'd23; logic [8:0] sum;
  assign sum = a + b;
  initial begin
    #1;
    if (sum == 9'd42) $display("PASS");
    else begin $display("FAIL sum=%0d", sum); $finish(1); end
    $finish;
  end
endmodule
