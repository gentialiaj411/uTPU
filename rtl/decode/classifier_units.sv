module decode_classifier(input logic clk,rst,start,input logic signed [31:0] hidden,weight,output logic busy,done,output logic signed [31:0] logit,output logic [31:0] argmax);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;logit<=0;argmax<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin logit<=hidden*weight;argmax<=0;busy<=0;done<=1;end end end
endmodule
module decode_embedding_row(input logic clk,rst,start,input logic signed [31:0] table_value,output logic busy,done,output logic signed [31:0] value);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin value<=table_value;busy<=0;done<=1;end end end
endmodule
