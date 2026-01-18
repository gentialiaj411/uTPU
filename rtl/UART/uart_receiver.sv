/*
 * Module `uart_receiver`
 *
 * This is an 8 bit uart receiver module.
 * The 8 bit output from the transaction is given.
 */


`timescale 1ns/1ps

module uart_receiver #(
	parameter UART_BITS_TRANSFERED = 8,
	parameter OVERSAMPLE = 16
    ) (
        input  logic clk, rst, rx,
	output logic valid,
        output logic [UART_BITS_TRANSFERED-1:0] result
    );

    // State Machine Logic 

    typedef enum logic [1:0] {
        IDLE,
        START,
        DATA,
        STOP
    } state_e;

    state_e current_state;
    int received_bit = 0;
    int sample_count = 0;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= IDLE;
	    valid <= 1'b0;
	    received_bit <= 0;
	    sample_count <= 0;
        end else begin
	    valid <= 1'b0;
            case (current_state)
                IDLE: begin
                    if (~rx) begin
			sample_count <= OVERSAMPLE / 2;
                        current_state <= START;
                    end
                end
                START: begin
                    if (sample_count == 0) begin
			if (~rx) begin
			    received_bit <= 0;
			    sample_count <= OVERSAMPLE;
			    current_state <= DATA;
			end else begin
			    current_state <= IDLE;
			end
                    end else begin
			sample_count <= sample_count - 1;
		    end
                end
                DATA: begin
                    if (sample_count == 0) begin
			result[received_bit] <= rx;
			received_bit <= received_bit + 1;
			sample_count <= OVERSAMPLE;
			if (received_bit == UART_BITS_TRANSFERED-1)
			    current_state <= STOP;
                    end else begin
			sample_count <= sample_count - 1;
		    end
                end
                STOP: begin
                    if (sample_count == 0) begin
			if (rx) begin
			    valid <= 1'b1;
			end
			current_state <= IDLE;
		    end else begin
			sample_count <= sample_count - 1;
		    end
                end
            endcase
        end
    end

endmodule: uart_receiver
