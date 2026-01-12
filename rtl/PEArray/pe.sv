module pe #(
	parameter COMPUTE_DATA_WIDTH = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16
    ) (
	input logic clk, rst, compute, load_en,
	input logic [COMPUTE_DATA_WIDTH-1:0] in,
	input logic [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_in,
	//output logic [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_out,
	output logic [COMPUTE_DATA_WIDTH-1:0] activation_out,
	output logic [ACCUMULATOR_DATA_WIDTH-1:0] accumulator
    );

    logic [COMPUTE_DATA_WIDTH-1:0] weight;

    always_ff @(posedge clk) begin
        if (rst) begin
            weight <= '0;
            activation_out <= '0;
            //partial_sum_out <= '0;
            accumulator <= '0;
        end else begin
            activation_out <= in;
            if (load_en) 
            weight <= in;
            else if (compute) begin
            accumulator <= partial_sum_in + (in * weight);
            //partial_sum_out <= partial_sum_in + (data_in * weight);
           end
        end
    end
endmodule: pe

