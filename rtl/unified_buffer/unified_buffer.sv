
`timescale 1ns/1ps

module unified_buffer #(
	parameter BUFFER_SIZE 	     = 1024, // The amount of words in the buffer
	parameter BUFFER_WORD_SIZE   = 16,   // Number of bits stored in each cell
	parameter FIFO_DATA_WIDTH    = 8,    // Number of bits recieved/sent from/to fifos
	parameter COMPUTE_DATA_WIDTH = 4,  // Number of bits recieved/sent from/to compute unit
	parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE),
	parameter ARRAY_SIZE         = 8,
	parameter NUM_COMPUTE_LANES  = ARRAY_SIZE*ARRAY_SIZE,
	parameter STORE_DATA_WIDTH   = 16
    ) (
	input  logic clk, we, re, compute_en, fifo_en, store_en,
	output logic 			      done,
	input  logic 			      section,  // Used for fifo where 0 top/1 bot
	input  logic [ADDRESS_SIZE-1:0]	      address,
	input  logic [FIFO_DATA_WIDTH-1:0]    fifo_in,
	output logic [FIFO_DATA_WIDTH-1:0]    fifo_out,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0] compute_in [NUM_COMPUTE_LANES-1:0], 
	output logic signed [COMPUTE_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0],
	input  logic [STORE_DATA_WIDTH-1:0]   store_in,
	output logic [STORE_DATA_WIDTH-1:0]   store_out
    ); 
    
    localparam int ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;
    localparam int BANKS         = NUM_COMPUTE_LANES/ITEMS_IN_SLOT;
    localparam int BANK_BITS     = $clog2(BANKS);
    localparam int BANK_DEPTH    = BUFFER_SIZE / BANKS;
    localparam int BANK_ADDR_W   = $clog2(BANK_DEPTH);

    // Banked BRAM to provide enough read/write bandwidth for compute lanes.
    (* ram_style = "block" *) logic [BUFFER_WORD_SIZE-1:0] mem [BANKS-1:0][BANK_DEPTH-1:0];
    logic [BUFFER_WORD_SIZE-1:0] compute_word [BANKS-1:0];
    logic [BANK_BITS-1:0]        base_bank;
    logic [BANK_ADDR_W-1:0]      base_row;

    function automatic [BANK_ADDR_W-1:0] row_for_bank(
        input int bank_idx,
        input [BANK_ADDR_W-1:0] row,
        input [BANK_BITS-1:0] bank_sel
    );
        if (bank_idx < bank_sel)
            row_for_bank = row + 1'b1;
        else
            row_for_bank = row;
    endfunction

    always_comb begin
        base_bank = address[BANK_BITS-1:0];
        base_row  = address[ADDRESS_SIZE-1:BANK_BITS];
    end

    genvar gi, gj;
    generate
        for (gi = 0; gi < BANKS; gi++) begin: gen_pack
            for (gj = 0; gj < ITEMS_IN_SLOT; gj++) begin: gen_pack_lanes
                assign compute_word[gi][(COMPUTE_DATA_WIDTH*gj) +: COMPUTE_DATA_WIDTH] =
                    compute_in[gj + gi*ITEMS_IN_SLOT];
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        done <= we | re;

        if (we) begin
            if (compute_en) begin
                // write compute_in lanes into banked memory words starting at address
                for (int i = 0; i < BANKS; i++) begin
                    mem[i][row_for_bank(i, base_row, base_bank)] <= compute_word[i];
                end
            end else if (fifo_en) begin
                case (section)
                    1'b0: mem[base_bank][base_row][0 +: FIFO_DATA_WIDTH] <= fifo_in;                // low byte
                    1'b1: mem[base_bank][base_row][FIFO_DATA_WIDTH +: FIFO_DATA_WIDTH] <= fifo_in;  // high byte
                endcase
	    end else if (store_en) begin 
		mem[base_bank][base_row][STORE_DATA_WIDTH-1:0] <= store_in;
	    end

        end else if (re) begin
            if (compute_en) begin
                // read banked memory words into compute_out lanes (1-cycle read latency)
                for (int i = 0; i < BANKS; i++) begin
                    for (int j = 0; j < ITEMS_IN_SLOT; j++) begin
                        compute_out[j + i*ITEMS_IN_SLOT]
                            <= mem[i][row_for_bank(i, base_row, base_bank)][(COMPUTE_DATA_WIDTH*j) +: COMPUTE_DATA_WIDTH];
                    end
                end
            end else if (fifo_en) begin
                case (section)
                    1'b0: fifo_out <= mem[base_bank][base_row][0 +: FIFO_DATA_WIDTH];
                    1'b1: fifo_out <= mem[base_bank][base_row][FIFO_DATA_WIDTH +: FIFO_DATA_WIDTH];
                endcase
	    end else if (store_en) begin
		store_out <= mem[base_bank][base_row][STORE_DATA_WIDTH-1:0];
	    end
        end
    end

endmodule: unified_buffer
