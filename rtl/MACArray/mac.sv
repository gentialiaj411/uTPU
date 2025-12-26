module mac #(
	parameter COMPUTE_DATA_WIDTH  	 = 4, // int4 systolic array input width
	parameter ACCUMULATOR_DATA_WIDTH = 16
    ) (
	input  logic clk, rst, compute, load_en,
	input  logic [COMPUTE_DATA_WIDTH-1:0]     in,
	output logic [ACCUMULATOR_DATA_WIDTH-1:0] accumulator 
    );
   
    logic [COMPUTE_DATA_WIDTH-1:0] weight;
    always_ff @(posedge clk) begin
	if (rst)
	    weight <= 4'0;
	else if (load_en) 
	    weight <= in;
	else if (compute)
	    accumulator <= accumulator + (in * weight);
    end	

endmodule: mac
