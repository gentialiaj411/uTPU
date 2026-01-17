
`timescale 1ns/1ps

module unified_buffer #(
	parameter BUFFER_SIZE 	     = 1024, // The amount of words in the buffer
	parameter BUFFER_WORD_SIZE   = 16,   // Number of bits stored in each cell
	parameter FIFO_DATA_WIDTH    = 8,    // Number of bits recieved/sent from/to fifos
	parameter COMPUTE_DATA_WIDTH = 4,  // Number of bits recieved/sent from/to compute unit
	parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE),
	parameter ARRAY_SIZE         = 8,
	parameter NUM_COMPUTE_LANES  = ARRAY_SIZE*ARRAY_SIZE
    ) (
	input  logic clk, we, re, compute_en, fifo_en,
	output logic 			      done,
	input  logic 			      section,  // Used for fifo where 0 top/1 bot
	input  logic [ADDRESS_SIZE-1:0]	      address,
	input  logic [FIFO_DATA_WIDTH-1:0]    fifo_in,
	output logic [FIFO_DATA_WIDTH-1:0]    fifo_out,
	input  logic [COMPUTE_DATA_WIDTH-1:0] compute_in [NUM_COMPUTE_LANES-1:0], 
	output logic [COMPUTE_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0] 
    ); 
    
    localparam int ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;

    logic [BUFFER_WORD_SIZE-1:0] mem [BUFFER_SIZE-1:0];

    always_ff @(posedge clk) begin
        done <= 1'b0;

        if (we) begin
            if (compute_en) begin
                // write compute_in lanes into memory words starting at address
                for (int i = 0; i < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); i++) begin
                    for (int j = 0; j < ITEMS_IN_SLOT; j++) begin
                        // Use variable part-select: base +: width
                        mem[address+i][(COMPUTE_DATA_WIDTH*j) +: COMPUTE_DATA_WIDTH]
                            <= compute_in[j + i*ITEMS_IN_SLOT];
                    end
                end
            end else if (fifo_en) begin
                case (section)
                    1'b0: mem[address][0 +: FIFO_DATA_WIDTH] <= fifo_in;                // low byte
                    1'b1: mem[address][FIFO_DATA_WIDTH +: FIFO_DATA_WIDTH] <= fifo_in;  // high byte
                endcase
            end
            done <= 1'b1;

        end else if (re) begin
            if (compute_en) begin
                // read memory words into compute_out lanes
                for (int i = 0; i < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); i++) begin
                    for (int j = 0; j < ITEMS_IN_SLOT; j++) begin
                        compute_out[j + i*ITEMS_IN_SLOT]
                            <= mem[address+i][(COMPUTE_DATA_WIDTH*j) +: COMPUTE_DATA_WIDTH];
                    end
                end
            end else if (fifo_en) begin
                case (section)
                    1'b0: fifo_out <= mem[address][0 +: FIFO_DATA_WIDTH];
                    1'b1: fifo_out <= mem[address][FIFO_DATA_WIDTH +: FIFO_DATA_WIDTH];
                endcase
            end
            done <= 1'b1;
        end
    end


endmodule: unified_buffer
