`timescale 1ns/1ps

module clk_divider #(
    	parameter INPUT_CLK            = 27000000,
	parameter UART_CLK             = 1000000
    ) (
	input  logic clk, rst,
	output logic uart_clk
    );

    localparam DIVIDER_COUNT = INPUT_CLK / UART_CLK;

    
    logic [$clog2(DIVIDER_COUNT)-1:0] count;


    always_ff @(posedge clk) begin
	if (rst) begin
	    count <= '0;
	    uart_clk <= 0;
	end else begin
	    count <= count + 1'b1;
	    uart_clk <= 1'b0;
	    if (count == DIVIDER_COUNT - 1) begin
		uart_clk <= 1'b1;
		count <= 0;
	    end
	end
    end

endmodule: clk_divider
