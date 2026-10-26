module decode_stream_reader #(parameter DEPTH=4096)(input logic clk,rst,input logic [7:0] in_byte,input logic in_valid,output logic in_ready,output logic [7:0] out_byte,output logic out_valid,input logic out_ready);
 logic [7:0] mem[0:DEPTH-1]; logic [12:0] wptr,rptr,count; assign in_ready=(count<DEPTH); assign out_valid=(count!=0); assign out_byte=mem[rptr[11:0]];
 always_ff @(posedge clk) begin if(rst) begin wptr<=0;rptr<=0;count<=0;end else begin if(in_valid&&in_ready) begin mem[wptr[11:0]]<=in_byte;wptr<=wptr+1;end if(out_valid&&out_ready) rptr<=rptr+1; case({in_valid&&in_ready,out_valid&&out_ready}) 2'b10:count<=count+1;2'b01:count<=count-1;default:count<=count;endcase end end
endmodule
module top_decode #(parameter MAX_SEQ=128)(input logic clk,rst,input logic [7:0] flash_data,input logic flash_valid,output logic flash_ready,input logic consume,output logic [7:0] stream_data,output logic stream_valid);
 decode_stream_reader reader(clk,rst,flash_data,flash_valid,flash_ready,stream_data,stream_valid,consume);
endmodule
