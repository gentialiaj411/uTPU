module unified_buffer #(
	parameter BUFFER_SIZE 	     = 1024, // The amount of words in the buffer
	parameter BUFFER_WORD_SIZE   = 16,   // Number of bits stored in each cell
	parameter FIFO_DATA_WIDTH    = 8,    // Number of bits recieved/sent from/to fifos
	parameter COMPUTE_DATA_WIDTH = 4,  // Number of bits recieved/sent from/to compute unit
	parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE),
	parameter ARRAY_SIZE         = 2,
	parameter NUM_COMPUTE_LANES  = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH
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
    
    logic [BUFFER_WORD_SIZE-1:0] mem [BUFFER_SIZE-1:0];

    always_ff @(posedge clk) begin
	done <= 1'b0;
	if (we) begin
	    if (compute_en) begin
		for (int i=0; i < NUM_COMPUTE_LANES; i++) begin 
		    mem[address][COMPUTE_DATA_WIDTH*(i+1)-1:COMPUTE_DATA_WIDTH*i] <= compute_in[i];
		end
	    end else if (fifo_en)
		case (section)
		    1'b0: mem[address][FIFO_DATA_WIDTH-1:0] <= fifo_in;
		    1'b1: mem[address][FIFO_DATA_WIDTH*2-1:FIFO_DATA_WIDTH] <= fifo_in;
		endcase
	    done <= 1'b1;
	end else if (re) begin
	    if (compute_en) begin
		for (int i=0; i < NUM_COMPUTE_LANES; i++) begin
		    compute_out[i] <= mem[address][COMPUTE_DATA_WIDTH*(i+1)-1:COMPUTE_DATA_WIDTH*i];
		end
	    end else if (fifo_en)
		case (section)
		    1'b0: fifo_out <= mem[address][FIFO_DATA_WIDTH-1:0];
		    1'b1: fifo_out <= mem[address][FIFO_DATA_WIDTH*2-1:FIFO_DATA_WIDTH];
    		endcase
	    done <= 1'b1;
	end
    end


endmodule: unified_buffer
