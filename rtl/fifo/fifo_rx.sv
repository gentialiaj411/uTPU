
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
    logic read_ok_d;

    
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
	    read_ok_d <= 1'b0;
	end else begin 
	    read_ok_d <= read_ok;
	    r_valid <= read_ok_d;
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
