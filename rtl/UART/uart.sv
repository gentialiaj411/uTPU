`include "uart_transmitter.sv"
`include "uart_receiver.sv"

module uart #(
	parameter UART_BITS_TRANSFERED = 8,
	parameter INPUT_CLK            = 27000000,
	parameter UART_CLK             = 1000000
    ) (
	input  logic clk, rst, tx_start, rx,
	output logic rx_valid, tx,
	input  logic [UART_BITS_TRANSFERED-1:0] tx_message
	output logic [UART_BITS_TRANSFERED-1:0] rx_result,
    );

    logic uart_clk;

    clk_divider #(
	.INPUT_CLK(INPUT_CLK),
	.UART_CLK(UART_CLK)
    ) u_clk_divider (
	.clk(clk),
	.rst(rst),
	.uart_clk(uart_clk)
    );

    uart_receiver #(
	.UART_BITS_TRANSFERED(UART_BITS_TRANSFERED)
    ) u_uart_receiver (
	.clk(uart_clk),
	.rst(rst),
	.rx(rx),
	.valid(rx_valid),
	.result(rx_result)
    );

    uart_transmitter #(
	.UART_BITS_TRANSFERED(UART_BITS_TRANSFERED)
    ) u_uart_transmitter (
	.clk(uart_clk),
	.rst(rst),
	.start(tx_start),
	.tx(tx),
	.message(tx_message)
    );

endmodule: uart
