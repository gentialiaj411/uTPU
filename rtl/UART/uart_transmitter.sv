module uart_transmitter (
	input  logic 	   rst,
	input  logic       clk,
	output logic       tx,
	input  logic 	   start,
	input  logic [7:0] message
    );

    localparam int BITS_TRANSFERED = 8;

    typedef enum logic {
	IDLE,
	START,
	MESSAGE,
	STOP
    } state_e;

    state_e current_state;
    int transmitting_bit = 0;

    always_ff @(posedge clk, posedge rst) begin
	if (rst)
	    current_state <= IDLE;
	else begin
	    case (current_state)
		IDLE: begin
		    tx <= 1'b1;
		    current_state <= (start) ? START : IDLE;
		end
		START: begin
		    tx <= 1'b0;
		    current_state <= MESSAGE;
		end
		MESSAGE: begin
		    tx <= message[transmitting_bit];
		    transmitting_bit <= transmitting_bit + 1;
		    if (transmitting_bit == BITS_TRANSFERED) begin
			transmitting_bit <= 0;
			current_state <= STOP;
		    end
		end
		STOP: begin
		    tx <= 1'b1;
		    current_state <= IDLE;
		end
	    endcase
	end  
    end

endmodule: uart_transmitter
