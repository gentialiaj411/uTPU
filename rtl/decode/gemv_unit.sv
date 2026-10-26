module decode_gemv_unit #(parameter LANES=16)(
 input logic clk,input logic rst,input logic start,input logic valid,input logic last,
 input logic int4_mode,input logic signed [7:0] activation_code,input logic signed [7:0] weight_code,
 output logic ready,output logic done,output logic signed [31:0] result);
 logic signed [31:0] acc; logic busy;
 assign ready=busy;
 always_ff @(posedge clk) begin
   if(rst) begin acc<=0;busy<=0;done<=0;result<=0; end
   else begin done<=0;
     if(start&&!busy) begin busy<=1;acc<=0;end
     else if(busy&&valid) begin
       acc <= acc + activation_code * (int4_mode ? $signed({{4{weight_code[3]}},weight_code[3:0]}) : weight_code);
       if(last) begin result<=acc + activation_code * (int4_mode ? $signed({{4{weight_code[3]}},weight_code[3:0]}) : weight_code); done<=1;busy<=0; end
     end
   end
 end
endmodule
