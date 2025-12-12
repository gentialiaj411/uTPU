module quantizer (
	input logic [15:0] in_val,
	output logic [3:0]  out_val
    );
    

    assign out_val = in_val >> 12;

endmodule
