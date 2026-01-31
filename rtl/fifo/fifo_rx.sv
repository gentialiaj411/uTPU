
`timescale 1ns/1ps

module fifo_rx #(
	parameter FIFO_WIDTH = 256,
	parameter FIFO_DATA_WIDTH = 8
    ) (
	input  logic clk, rst, we, re, valid,
	output logic empty, full,
	input  logic [FIFO_DATA_WIDTH-1:0] w_data,
	output logic [FIFO_DATA_WIDTH-1:0] r_data,
	output logic r_valid
    );
    
    localparam POINTER_WIDTH = $clog2(FIFO_WIDTH);

    // Prefer BRAM over LUTs for FIFO storage.
    (* ram_style = "block" *) logic [FIFO_DATA_WIDTH-1:0] mem [FIFO_WIDTH-1:0];

    logic write_ok, read_ok;
    logic [POINTER_WIDTH:0] w_ptr, r_ptr; // Read and write pointers with extra MSB (Cummings 2002)

    
    assign empty = (w_ptr == r_ptr);
    assign full  = (w_ptr == {~r_ptr[POINTER_WIDTH], r_ptr[POINTER_WIDTH-1:0]});

    assign write_ok = we && !full;
    assign read_ok  = re && !empty;

    always_ff @(posedge clk) begin
	if (rst) begin
	    w_ptr   <= 0;
	    r_ptr   <= 0;
	    r_data  <= 0;
	    r_valid <= 1'b0;
	end else begin 
	    // BRAM has 1-cycle read latency: r_data is valid on the cycle AFTER read_ok.
	    // r_valid should go high exactly when r_data becomes valid (1 cycle after read_ok).
	    r_valid <= read_ok;  // Delay by 1 cycle to match r_data timing
	    if (write_ok) begin
		mem[w_ptr[POINTER_WIDTH-1:0]] <= w_data;
		w_ptr <= w_ptr + 1'b1;
	    end
	    if (read_ok) begin
		r_data <= mem[r_ptr[POINTER_WIDTH-1:0]];
		r_ptr  <= r_ptr + 1'b1;
	    end
	end
    end
    
endmodule: fifo_rx
