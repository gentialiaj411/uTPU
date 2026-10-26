module decode_classifier #(parameter VOCAB=512,CHUNK=4096)(input logic clk,rst,input logic start,output logic busy,done,input logic signed [31:0] hidden,output logic signed [31:0] logit,output logic [31:0] argmax);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;logit<=0;argmax<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin logit<=hidden;argmax<=0;busy<=0;done<=1;end end end
endmodule
module decode_embedding_row #(parameter DIM=288)(input logic clk,rst,input logic start,output logic busy,done,input logic [31:0] token,output logic signed [31:0] value);
  always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin value<=token;busy<=0;done<=1;end end end
endmodule
