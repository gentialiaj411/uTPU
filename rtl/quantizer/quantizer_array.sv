`timescale 1ns/1ps

module quantizer_array #(
	parameter QUANTIZER_SIZE         = 8*8,
	parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE),
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter int QUANTIZER_PIPE_DEPTH = 0
    ) (
	input  logic                                      clk,
	input  logic                                      rst,
	input  logic [(QUANTIZER_SIZE*ACCUMULATOR_DATA_WIDTH)-1:0] ins_flat,
	input  logic                                      requant_enable,
	input  logic [(QUANTIZER_SIZE*16)-1:0]            requant_multiplier_flat,
	input  logic [(QUANTIZER_SIZE*16)-1:0]            requant_right_shift_flat,
	output logic [(QUANTIZER_SIZE*COMPUTE_DATA_WIDTH)-1:0] results_flat
    );

    genvar i;
    generate 
	for (i = 0; i < QUANTIZER_SIZE; i++) begin: create_array
	    localparam int IN_LO = i * ACCUMULATOR_DATA_WIDTH;
	    localparam int OUT_LO = i * COMPUTE_DATA_WIDTH;
	    quantizer #(
		.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
		.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
		.QUANTIZER_PIPE_DEPTH(QUANTIZER_PIPE_DEPTH)
	    ) u_quant (
		.clk(clk),
		.rst(rst),
		.in(ins_flat[IN_LO +: ACCUMULATOR_DATA_WIDTH]),
		.requant_enable(requant_enable),
		.requant_multiplier(requant_multiplier_flat[(i * 16) +: 16]),
		.requant_right_shift(requant_right_shift_flat[(i * 16) +: 16]),
		.result(results_flat[OUT_LO +: COMPUTE_DATA_WIDTH])
	    );
	end
    endgenerate

endmodule: quantizer_array
