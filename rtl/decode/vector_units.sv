module decode_act_quant(input logic clk,rst,start,input logic signed [31:0] value,input logic [31:0] reciprocal,output logic busy,done,output logic signed [7:0] code);
 logic signed [63:0] product;
 always_comb product=value*$signed(reciprocal);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;code<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin if(product>>>24>127) code<=127; else if(product>>>24 < -128) code<=-128; else code <= $signed(product>>>24);busy<=0;done<=1;end end end
endmodule
module decode_rmsnorm(input logic clk,rst,start,input logic signed [31:0] value,gamma,output logic busy,done,output logic signed [31:0] out_value);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value <= (value*gamma)>>>12;busy<=0;done<=1;end end end
endmodule
module decode_rope(input logic clk,rst,start,input logic signed [31:0] x,y,c,s,output logic busy,done,output logic signed [31:0] out_x,out_y);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_x<=0;out_y<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_x <= ((x*c)-(y*s))>>>15;out_y <= ((x*s)+(y*c))>>>15;busy<=0;done<=1;end end end
endmodule
module decode_silu_mul(input logic clk,rst,start,input logic signed [31:0] silu,w3,output logic busy,done,output logic signed [31:0] out_value);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value<=silu*w3;busy<=0;done<=1;end end end
endmodule
module decode_residual(input logic clk,rst,start,input logic signed [31:0] a,b,output logic busy,done,output logic signed [31:0] out_value);
 always_ff @(posedge clk) begin if(rst) begin busy<=0;done<=0;out_value<=0;end else begin done<=0;if(start&&!busy) busy<=1;else if(busy) begin out_value<=a+b;busy<=0;done<=1;end end end
endmodule
