module decode_kv_cache #(parameter MAX_SEQ=128,KV_BITS=8,LAYERS=6,KV_HEADS=6,HEAD_SIZE=48)(input logic clk,rst,input logic start,output logic busy,done,input logic [15:0] position,input logic signed [7:0] key_in,value_in,output logic signed [7:0] key_out,value_out);
  logic signed [7:0] key_mem [0:MAX_SEQ-1]; logic signed [7:0] value_mem [0:MAX_SEQ-1];
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;key_out<=0;value_out<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin key_mem[position]<=key_in;value_mem[position]<=value_in;key_out<=key_mem[position];value_out<=value_mem[position];busy<=0;done<=1;end end end
endmodule
module decode_score(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] q,k, input logic [7:0] qmax,output logic signed [31:0] score);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;score<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin score<=q*k;busy<=0;done<=1;end end end
endmodule
module decode_softmax(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] in_value,output logic [15:0] probability);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;probability<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin probability<=in_value[15:0];busy<=0;done<=1;end end end
endmodule
module decode_attention_value(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] p,v,output logic signed [31:0] out_value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value<=p*v;busy<=0;done<=1;end end end
endmodule
