
`timescale 1ns/1ps

module uart_transmitter #(
	parameter UART_BITS_TRANSFERED = 8,
	parameter OVERSAMPLE = 16
    ) (
	input  logic clk, rst, start,
	output logic tx,
	input  logic [UART_BITS_TRANSFERED-1:0] message
    );

    typedef enum logic [1:0] {
	IDLE,
	START,
	MESSAGE,
	STOP
    } state_e;

    state_e current_state;
    int transmitting_bit = 0;
    int tick_count = 0;

    always_ff @(posedge clk, posedge rst) begin
	if (rst) begin
	    current_state <= IDLE;
	    transmitting_bit <= 0;
	    tick_count <= 0;
	    tx <= 1'b1;
	end else begin
	    case (current_state)
		IDLE: begin
		    tx <= 1'b1;
		    if (start) begin
			current_state <= START;
			tick_count <= OVERSAMPLE;
		    end
		end
		START: begin
		    tx <= 1'b0;
		    if (tick_count == 0) begin
			current_state <= MESSAGE;
			tick_count <= OVERSAMPLE;
			transmitting_bit <= 0;
		    end else begin
			tick_count <= tick_count - 1;
		    end
		end
		MESSAGE: begin
		    tx <= message[transmitting_bit];
		    if (tick_count == 0) begin
			if (transmitting_bit == UART_BITS_TRANSFERED-1) begin
			    current_state <= STOP;
			    tick_count <= OVERSAMPLE;
			    transmitting_bit <= 0;
			end else begin
			    transmitting_bit <= transmitting_bit + 1;
			    tick_count <= OVERSAMPLE;
			end
		    end else begin
			tick_count <= tick_count - 1;
		    end
		end
		STOP: begin
		    tx <= 1'b1;
		    if (tick_count == 0) begin
			current_state <= IDLE;
		    end else begin
			tick_count <= tick_count - 1;
		    end
		end
	    endcase
	end  
    end

endmodule: uart_transmitter
