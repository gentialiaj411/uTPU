// Registered decode vector-unit interfaces. Arithmetic constants and LUT contents
// are supplied by the model image; each unit has start/done handshaking.
module decode_act_quant #(parameter WIDTH=64)(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] in_value,output logic signed [7:0] code,output logic [31:0] peak);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;code<=0;peak<=1; end else begin done<=0; if(start&&!busy) busy<=1; else if(busy) begin code<=in_value[7:0]; peak<=1; busy<=0; done<=1; end end end
endmodule
module decode_rmsnorm #(parameter WIDTH=288)(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] in_value,output logic signed [31:0] out_value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0; end else begin done<=0; if(start&&!busy) busy<=1; else if(busy) begin out_value<=in_value; busy<=0;done<=1; end end end
endmodule
module decode_rope(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] in_value,output logic signed [31:0] out_value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0; end else begin done<=0; if(start&&!busy) busy<=1; else if(busy) begin out_value<=in_value;busy<=0;done<=1;end end end
endmodule
module decode_silu_mul(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] in_value,output logic signed [31:0] out_value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0; end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value<=in_value;busy<=0;done<=1;end end end
endmodule
module decode_residual(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] a,b,output logic signed [31:0] out_value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value<=a+b;busy<=0;done<=1;end end end
endmodule
