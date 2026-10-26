module decode_gemv_unit #(parameter LANES=16,parameter MAX_K=768)(
 input logic clk,rst,input logic start,input logic valid,input logic last,input logic int4_mode,
 input logic signed [7:0] activation_code,input logic [LANES*8-1:0] weight_bytes,
 output logic ready,output logic done,output logic signed [LANES*32-1:0] result);
 logic busy; logic [15:0] k_count; logic signed [31:0] accum[0:LANES-1]; logic signed [LANES*32-1:0] result_int; integer i; logic signed [7:0] w;
 assign ready=busy;
`ifdef GEMV_MUTATE
 assign result = result_int ^ ({{(LANES*32-33){1'b0}},1'b1,32'b0});
`else
 assign result = result_int;
`endif
 always_ff @(posedge clk) begin
   if(rst) begin busy<=0;done<=0;k_count<=0; result_int<=0; for(i=0;i<LANES;i=i+1) accum[i]<=0; end
   else begin done<=0;
     if(start&&!busy) begin busy<=1;k_count<=0;for(i=0;i<LANES;i=i+1) accum[i]<=0;end
     else if(busy&&valid) begin
       for(i=0;i<LANES;i=i+1) begin
         if(int4_mode) w = (k_count[0] == 1'b0) ? $signed({{4{weight_bytes[i*8+3]}},weight_bytes[i*8 +: 4]}) : $signed({{4{weight_bytes[i*8+7]}},weight_bytes[i*8+4 +: 4]});
         else w = $signed(weight_bytes[i*8 +: 8]);
         accum[i] <= accum[i] + activation_code*w;
         if(last) result_int[i*32 +: 32] <= accum[i] + activation_code*w;
       end
       k_count<=k_count+1;
       if(last) begin busy<=0;done<=1;end
     end
   end
 end
endmodule
