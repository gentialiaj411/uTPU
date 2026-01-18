
`timescale 1ns/1ps

module uart #(
	parameter UART_BITS_TRANSFERED = 8,
	parameter INPUT_CLK            = 27000000,
	parameter UART_CLK             = 1000000
    ) (
	input  logic clk, rst, tx_start, rx,
	output logic rx_valid, tx,
	input  logic [UART_BITS_TRANSFERED-1:0] tx_message,
	output logic [UART_BITS_TRANSFERED-1:0] rx_result
    );

    logic uart_clk;
    logic rx_valid_uart;
    logic [UART_BITS_TRANSFERED-1:0] rx_result_uart;
    logic rx_toggle_uart;
    logic [UART_BITS_TRANSFERED-1:0] rx_data_hold;
    logic rx_toggle_sync1, rx_toggle_sync2;
    logic rx_pending_valid;

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
	.valid(rx_valid_uart),
	.result(rx_result_uart)
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

    // Capture RX data in uart_clk domain and signal via toggle.
    always_ff @(posedge uart_clk or posedge rst) begin
	if (rst) begin
	    rx_toggle_uart <= 1'b0;
	    rx_data_hold   <= '0;
	end else if (rx_valid_uart) begin
	    rx_data_hold   <= rx_result_uart;
	    rx_toggle_uart <= ~rx_toggle_uart;
	end
    end

    // Sync toggle into clk domain and generate a 1-cycle pulse.
    always_ff @(posedge clk or posedge rst) begin
	if (rst) begin
	    rx_toggle_sync1 <= 1'b0;
	    rx_toggle_sync2 <= 1'b0;
	    rx_valid        <= 1'b0;
	    rx_result       <= '0;
	    rx_pending_valid <= 1'b0;
	end else begin
	    rx_toggle_sync1 <= rx_toggle_uart;
	    rx_toggle_sync2 <= rx_toggle_sync1;
	    if (rx_toggle_sync1 ^ rx_toggle_sync2) begin
		rx_result <= rx_data_hold;
		rx_pending_valid <= 1'b1;
	    end else if (rx_pending_valid) begin
		rx_pending_valid <= 1'b0;
	    end
	    rx_valid <= rx_pending_valid;
	end
    end

endmodule: uart
