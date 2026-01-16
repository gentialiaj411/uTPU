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
    
    localparam ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;

    logic [BUFFER_WORD_SIZE-1:0] mem [BUFFER_SIZE-1:0];

    always_ff @(posedge clk) begin
	done <= 1'b0;
	if (we) begin
	    if (compute_en) begin
		for (int i=0; i < NUM_COMPUTE_LANES/ITEMS_IN_SLOT; i++) begin 
		    for (int j=0; j < ITEM_IN_SLOT; j++) begin
			mem[address+i][COMPUTE_DATA_WIDTH*(j+1)-1:COMPUTE_DATA_WIDTH*j] 
			         <= compute_in[j+i*ITEMS_IN_SLOT];
		    end
		end
	    end else if (fifo_en)
		case (section)
		    1'b0: mem[address][FIFO_DATA_WIDTH-1:0] <= fifo_in;
		    1'b1: mem[address][FIFO_DATA_WIDTH*2-1:FIFO_DATA_WIDTH] <= fifo_in;
		endcase
	    done <= 1'b1;
	end else if (re) begin
	    if (compute_en) begin
		for (int i=0; i < NUM_COMPUTE_LANES/ITEMS_IN_SLOT; i++) begin 
		    for (int j=0; j < ITEM_IN_SLOT; j++) begin
			compute_in[j+i*ITEMS_IN_SLOT] 
			       <= mem[address+i][COMPUTE_DATA_WIDTH*(j+1)-1:COMPUTE_DATA_WIDTH*j];
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
