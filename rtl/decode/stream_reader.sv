module decode_stream_reader #(parameter FIFO_BYTES=4096)(input logic clk,rst,input logic [7:0] flash_data,input logic flash_valid,output logic flash_ready,input logic consume,output logic [7:0] data,output logic valid);
  logic [7:0] fifo [0:FIFO_BYTES-1]; logic [$clog2(FIFO_BYTES):0] wr,rd,count;
  assign flash_ready=(count<FIFO_BYTES); assign valid=(count!=0); assign data=fifo[rd[$clog2(FIFO_BYTES)-1:0]];
  always_ff @(posedge clk) begin if(rst) begin wr<=0;rd<=0;count<=0;end else begin if(flash_valid&&flash_ready) begin fifo[wr[$clog2(FIFO_BYTES)-1:0]]<=flash_data;wr<=wr+1;end if(consume&&valid) rd<=rd+1; case({flash_valid&&flash_ready,consume&&valid}) 2'b10:count<=count+1;2'b01:count<=count-1;default:count<=count;endcase end end
endmodule
