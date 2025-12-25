


module top #(
	parameter ALPHA			 = 2,
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16, 
	parameter ARRAY_SIZE		 = 2,
	parameter FIFO_WIDTH		 = 256,
	parameter FIFO_DATA_WIDTH	 = 8,
	parameter BUFFER_SIZE		 = 1024,
	parameter BUFFER_WORD_SIZE	 = 16,
	parameter ADDRESS_SIZE		 = $clog2(BUFFER_SIZE)

    ) (
	input  logic clk, rst, start,
	input  logic rx,
	output logic tx
    );

    // Controller registers
    logic [ADDRESS_SIZE-1:0] address;
    logic 		     relu_en;
    
    // FIFO reciever control signals/flags
    logic rx_we, rx_re, rx_empty, rx_full, rx_valid;
    // FIFO reciever data
    logic [FIFO_DATA_WIDTH-1:0] rx_to_fifo;
    logic [FIFO_DATA_WIDTH-1:0] rx_fifo_to_mem;


    // FIFO transmitter control signals/flags
    logic tx_we, tx_re, tx_empty, tx_full, tx_start;
    // FIFO transmitter data
    logic [FIFO_DATA_WIDTH-1:0] tx_to_fifo;
    logic [FIFO_DATA_WIDTH-1:0] mem_to_tx_fifo;


    // MAC Array control signals/flags
    logic compute_start, compute_load_en;
    // MAC Array data
    logic [COMPUTE_DATA_WIDTH-1:0]     mem_to_compute;
    logic [ACCUMULATOR_DATA_WIDTH-1:0] compute_out;


    // Quantizer data
    logic [ACCUMULATOR_DATA_WIDTH-1:0] accumulator_in;
    logic [COMPUTE_DATA_WIDTH-1:0]     accumulator_out;


    // ReLU data
    logic [COMPUTE_DATA_WIDTH-1:0] relu_in;
    logic [COMPUTE_DATA_WIDTH-1:0] relu_out;


    // Buffer control signals/flags
    logic buffer_we, buffer_re, buffer_compute_en, buffer_fifo_en;
    // Buffer data
    logic [COMPUTED_DATA_WIDTH-1:0] compute_to_buffer;


    uart_reciever reciever ( 
	    .rst(rst),
	    .clk(clk),
	    .rx(rx),
	    .valid(rx_valid),
	    .result(rx_to_fifo)
	);

    uart_transmitter transmitter (
	    .rst(rst),
	    .clk(clk),
	    .tx(tx),
	    .start(tx_start),
	    .message(tx_to_fifo)
	);

    fifo_rx fifo_in #(
	    .FIFO_WIDTH(FIFO_WIDTH),
	    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
	) (
	    .clk(clk),
	    .rst(rst),
	    .we(rx_we),	// These go to the controller
	    .re(rx_re),
	    .empty(rx_empty),
	    .full(rx_full),
	    .w_data(rx_to_fifo),		// These two go to memory
	    .r_data(rx_fifo_to_mem)

	);

    fifo_tx fifo_out #(
	    .FIFO_WIDTH(FIFO_WIDTH),
	    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
	) (
	    .clk(clk),
	    .rst(rst),
	    .we(tx_we),
	    .re(tx_re),
	    .start(tx_start),
	    .w_data(tx_to_fifo),
	    .r_data(mem_to_tx_fifo)
	);

    mac_array mac #(
	    .ARRAY_SIZE(ARRAY_SIZE),
	    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	    .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH)
	) (
	    .clk(clk),
	    .rst(rst),
	    .compute(compute_start),
	    .load_en(compute_load_en),
	    .in(mem_to_compute),
	    .accumulator(compute_out)
	);

    quantizer quant #(
	    .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
	) (
	    .in_val(accumulator_in),
	    .result(accumulator_out)
	);

    leaky_relu relu #
	    .ALPHA(ALPHA),
	    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
	) (
	    .in(relu_in),
	    .result(relu_out)
	);

    unified_buffer buffer #(
	    .BUFFER_SIZE(BUFFER_SIZE),
	    .BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
	    .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH),
	    .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	    .ADDRESS_SIZE(ADDRESS_SIZE)
	) (
	    .clk(clk),
	    .we(buffer_we),
	    .re(buffer_re),
	    .compute_en(buffer_compute_en),
	    .fifo_en(buffer_fifo_en),
	    .address(address),
	    .fifo_in(rx_fifo_to_mem),
	    .fifo_out(mem_to_tx_fifo),
	    .compute_in(mem_to_compute),
	    .compute_out(compute_to_buffer)
	);


    typedef enum logic [2:0] {
	RESET, // Resets all of the ptrs
	FETCH, // Gets the next 
	DECODE,
	LOAD_STREAM,
	COMPUTE,
	STORE_STREAM,
	HALT
    } state_e

    state_e current_state;

    logic [BUFFER_WORD_SIZE-1:0] instruction;
    logic 		         instruction_half;


    typedef enum logic [] {
	
	    
    } opcode_e

    typedef enum logic {
	FETCH_INSTRUCTION,
	FETCH_ADDRESS
    } fetch_mode_e

    always @(posedge clk) begin
	if (rst) 
	    current_state <= RESET;
	else begin
	    case (current_state)
		RESET: 
		    instruction_half <= 1'b0;
		    current_state <= FETCH;  // Assuming resest can happen in one clk cycle
		FETCH: begin
		    if (~rx_empty && ~instruction_half) begin
			rx_re <= 1'b1;
			instruction[FIFO_DATA_WIDTH-1:0] <= r_data;
			rx_re <= 1'b0;
			instruction_half <= 1'b1;
		    end else if (~rx_empty && instruction_half) begin
	    		rx_re <= 1'b1;
			instruction[BUFFER_DATA_WIDTH-1:0] <= r_data;
			rx_re <= 1'b0;
			instruction_half <= 1'b0;
			current_state <= DECODE;
		    end
		end
		DECODE: begin


	    endcase
	end
    end



endmodule: top
