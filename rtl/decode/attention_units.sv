module decode_kv_cache #(parameter MAX_SEQ=128)(input logic clk,rst,start,input logic [7:0] position,input logic signed [7:0] key_in,value_in,output logic busy,done,output logic signed [7:0] key_out,value_out);
 logic signed [7:0] km[0:MAX_SEQ-1],vm[0:MAX_SEQ-1];
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;key_out<=0;value_out<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin km[position]<=key_in;vm[position]<=value_in;key_out<=km[position];value_out<=vm[position];busy<=0;done<=1;end end end
endmodule
module decode_score(input logic clk,rst,start,input logic signed [31:0] q,k,output logic busy,done,output logic signed [31:0] score);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;score<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin score<=q*k;busy<=0;done<=1;end end end
endmodule
module decode_softmax(input logic clk,rst,start,input logic signed [31:0] score,output logic busy,done,output logic [15:0] probability);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;probability<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin probability<=score[15:0];busy<=0;done<=1;end end end
endmodule
module decode_attention_value(input logic clk,rst,start,input logic signed [31:0] probability,value,output logic busy,done,output logic signed [31:0] result);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;result<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin result<=probability*value;busy<=0;done<=1;end end end
endmodule
