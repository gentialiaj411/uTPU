module unified_buffer #(
	parameter BUFFER_SIZE 	     = 1024, // The amount of words in the buffer
	parameter BUFFER_WORD_SIZE   = 16,   // Number of bits stored in each cell
	parameter FIFO_DATA_WIDTH    = 8,    // Number of bits recieved/sent from/to fifos
	parameter COMPUTE_DATA_WIDTH = 4,  // Number of bits recieved/sent from/to compute unit
	parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE)
    ) (
	input  logic clk, we, re, compute_en, fifo_en,
	output logic 			      done,
	input  logic [ADDRESS_SIZE-1:0]	      address,
	input  logic [FIFO_DATA_WIDTH-1:0]    fifo_in,
	output logic [FIFO_DATA_WIDTH-1:0]    fifo_out,
	input  logic [COMPUTE_DATA_WIDTH-1:0] compute_in, 
	output logic [COMPUTE_DATA_WIDTH-1:0] compute_out	
    );
    
    logic [BUFFER_WORD_SIZE-1:0] mem [BUFFER_SIZE-1:0];

    always_ff @(posedge clk) begin
	done <= 1'b0;
	if (we) begin
	    if (compute_en) 
		mem[address] <= compute_in;
	    else if (fifo_en)
		mem[address] <= fifo_in;
	    done <= 1'b1;
	end else if (re) begin
	    if (compute_en)
		compute_out <= mem[address];
	    else if (fifo_en)
		fifo_out <= mem[address];
	    done <= 1'b1;
	end
    end


endmodule: unified_buffer
