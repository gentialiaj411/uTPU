/*
 * Module `uart_receiver`
 *
 * This is an 8 bit uart receiver module.
 * The 8 bit output from the transaction is given.
 */

module uart_receiver (
        input logic        rst,
        input logic        clk,
        input logic        rx,
	output logic       valid,
        output logic [7:0] result
    );

    localparam int BITS_TRANSFERED = 8;

    // State Machine Logic 

    typedef enum logic {
        IDLE,
        RECEIVING
    } state_e;

    state_e current_state;
    int received_bit = 0;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= IDLE;
        end
	else begin
            case (current_state) 
                IDLE: begin 
                    if (~rx) begin
                        current_state <= RECEIVING;
                    end
                end
                RECEIVING: begin
                    result[received_bit] <= rx;
                    received_bit <= received_bit + 1;
                    if (received_bit == BITS_TRANSFERED) begin
                        received_bit  <= 0;
                        current_state <= IDLE;
                    end
                end
            endcase 
        end
    end

endmodule: uart_receiver
