module decode_sequencer(input logic clk,rst,input logic start,output logic busy,done,output logic [7:0] opcode,output logic [15:0] index);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;opcode<=0;index<=0;end else begin done<=0;if(start&&!busy) begin busy<=1;index<=0;opcode<=1;end else if(busy) begin index<=index+1; if(index==16'hffff) begin busy<=0;done<=1;end end end end
endmodule
