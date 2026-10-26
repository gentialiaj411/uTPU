module top_decode #(parameter DIM=288,LAYERS=6,VOCAB=32000,MAX_SEQ=128,KV_BITS=4)(input logic clk,rst,input logic token_valid,output logic token_ready,input logic [31:0] token_in,output logic token_valid_out,output logic [31:0] token_out,input logic [7:0] flash_data,input logic flash_valid,output logic flash_ready);
  logic reader_valid,consume; logic [7:0] reader_data; logic seq_busy,seq_done; logic [7:0] opcode; logic [15:0] index;
  decode_stream_reader #(.FIFO_BYTES(4096)) reader(clk,rst,flash_data,flash_valid,flash_ready,consume,reader_data,reader_valid);
  decode_sequencer sequencer(clk,rst,token_valid&&token_ready,seq_busy,seq_done,opcode,index);
  assign token_ready=!seq_busy; assign consume=reader_valid&&seq_busy; assign token_valid_out=seq_done; assign token_out=token_in;
endmodule
