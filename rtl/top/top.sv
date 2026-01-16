module top #(
	parameter UART_BITS_TRANSFERED   = 8,
	parameter ALPHA			 = 2,
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16, 
	parameter ARRAY_SIZE		 = 2,
	parameter ARRAY_SIZE_WIDTH       = $clog2(ARRAY_SIZE),
	parameter FIFO_WIDTH		 = 256,
	parameter FIFO_DATA_WIDTH	 = 8,
	parameter BUFFER_SIZE		 = 512,
	parameter BUFFER_WORD_SIZE	 = 16,
	parameter ADDRESS_SIZE		 = $clog2(BUFFER_SIZE),
	parameter OPCODE_WIDTH	 	 = 3,
	parameter RELU_SIZE		 = 2,
	parameter RELU_SIZE_WIDTH        = $clog2(RELU_SIZE),
	parameter QUANTIZER_SIZE         = 2,
	parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE)
    ) (
	input  logic clk, rst, start,
	input  logic rx,
	output logic tx
    );

    // Controller registers
    logic [ADDRESS_SIZE-1:0]       address; 
    logic 			   compute_en;
    logic 	   		   quantizer_en;
    logic 		           relu_en;
    logic 	                   bot_mem;
    logic 		   	   mem_section; // used for fifo where 0 top / 1 bot
    logic [COMPUTE_DATA_WIDTH-1:0] store_val [BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH-1:0];    
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
    logic [COMPUTE_DATA_WIDTH-1:0]     compute_in [ARRAY_SIZE-1:0];
    logic [ACCUMULATOR_DATA_WIDTH-1:0] compute_out [ARRAY_SIZE-1:0];


    // Quantizer data
    logic [ACCUMULATOR_DATA_WIDTH-1:0] quantizer_in [QUANTIZER_SIZE-1:0];
    logic [COMPUTE_DATA_WIDTH-1:0]     quantizer_out [QUANTIZER_SIZE-1:0];


    // ReLU data
    logic [COMPUTE_DATA_WIDTH-1:0] relu_in [RELU_SIZE-1:0];
    logic [COMPUTE_DATA_WIDTH-1:0] relu_out [RELU_SIZE-1:0];


    // Buffer control signals/flags
    logic buffer_we, buffer_re, buffer_compute_en, buffer_fifo_en, buffer_done;
    // Buffer data
    logic [COMPUTE_DATA_WIDTH-1:0] mem_to_compute    [ARRAY_SIZE-1:0];
    logic [COMPUTE_DATA_WIDTH-1:0] compute_to_buffer [ARRAY_SIZE-1:0];


    uart #(
	.UART_BITS_TRANSFERED(UART_BITS_TRANSFERED)
    ) u_uart (
	.clk(clk),
	.rst(rst),
	.tx_start(tx_start),
	.rx(rx),
	.rx_valid(rx_valid),
	.tx(tx),
	.tx_message(tx_to_fifo),
	.rx_result(rx_to_fifo)
    );

    fifo_rx #(
	.FIFO_WIDTH(FIFO_WIDTH),
	.FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_in (
	.clk(clk),
	.rst(rst),
	.we(rx_we),	// These go to the controller
	.re(rx_re),
	.empty(rx_empty),
	.full(rx_full),
	.w_data(rx_to_fifo),			    
	.r_data(rx_fifo_to_mem)
    );

    fifo_tx #(
	.FIFO_WIDTH(FIFO_WIDTH),
	.FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_out (
	.clk(clk),
	.rst(rst),
	.we(tx_we),
	.re(tx_re),
	.start(tx_start),
	.empty(tx_empty),
	.full(tx_full),
	.w_data(mem_to_tx_fifo),
	.r_data(tx_to_fifo)
    );

    pe_controller #(
	.ARRAY_SIZE(ARRAY_SIZE),
	.ARRAY_SIZE_WIDTH(ARRAY_SIZE_WIDTH),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	.BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
	.NUM_COMPUTE_LANES(NUM_COMPUTE_LANES)
    ) u_pe_array (
	.clk(clk),
	.rst(rst),
	.compute(compute_start),
	.load_en(compute_load_en),
	.data_arr(compute_in),
	.weights_in(compute_in),
	.results(compute_out)
    );

    quantizer_array #(
	.QUANTIZER_SIZE(QUANTIZER_SIZE),
	.QUANTIZER_SIZE_WIDTH(QUANTIZER_SIZE_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
    ) u_quantizer_array (
	.ins(quantizer_in),
	.results(quantizer_out)
    );

    leaky_relu_array #(
	.RELU_SIZE(RELU_SIZE),
	.RELU_SIZE_WIDTH(RELU_SIZE_WIDTH),
	.ALPHA(ALPHA),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
    ) u_leaky_relu_array (
	.in(relu_in),
	.result(relu_out)
    );

    unified_buffer #(
	.BUFFER_SIZE(BUFFER_SIZE),
	.BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
	.FIFO_DATA_WIDTH(FIFO_DATA_WIDTH),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ADDRESS_SIZE(ADDRESS_SIZE),
	.ARRAY_SIZE(ARRAY_SIZE)
    ) u_unified_buffer (
	.clk(clk),
	.we(buffer_we),
	.re(buffer_re),
	.compute_en(buffer_compute_en),
	.fifo_en(buffer_fifo_en),
	.done(buffer_done),
	.section(section),
	.address(address),
	.fifo_in(rx_fifo_to_mem),
	.fifo_out(mem_to_tx_fifo),
	.compute_in(compute_to_buffer),
	.compute_out(mem_to_compute)
    );


    typedef enum logic [3:0] {
	RESET_STATE, // Resets all of the ptrs
	FETCH_FIFO_STATE, // Gets the next 
	DECODE_STATE,
	FETCH_ADDRESS_STATE,
	FETCH_BUFFER_STATE,
	LOAD_STATE,
	COMPUTE_STATE,
	STORE_STATE,
	HALT_STATE
    } state_e

    state_e current_state;
    state_e next_state;

    typedef enum logic [OPCODE_WIDTH-1:0] {
	STORE_OP,
	FETCH_OP,
	RUN_OP,
	LOAD_OP,
	HALT_OP,
	NOP    
    } opcode_e
    
    typedef enum logic {
	FETCH_INSTRUCTION,
	FETCH_ADDRESS //ONLY USED FOR STORES
    } fetch_mode_e

    typedef enum logic {


    logic [BUFFER_WORD_SIZE-1:0] instruction;
    logic 		         instruction_half;

    opcode_e opcode;
    assign opcode = instruction[OPCODE_WIDTH-1:0];

    fetch_mode_e fetch_mode;
    logic address_indicator;

    // NEXT STATE FSM
    always_ff @(posedge clk) begin
	if (rst)
	    next_state <= RESET_STATE;
	else begin
	    case (current_state)
		RESET_STATE:
		    next_state <= FETCH_FIFO_STATE; // Assuming reset can happen in one clk cycle
		FETCH_FIFO_STATE: 
		    if (~rx_empty && instruction_half) begin
			if (fetch_mode == FETCH_ADDRESS && address_indicator)
			    next_state <= FETCH_ADDRESS_STATE;
			else
			    next_state <= DECODE_STATE;
		    end
		DECODE_STATE:
		    case (opcode)
			STORE_OP: 
			    next_state <= FETCH_FIFO_STATE;
			FETCH_OP:
			    next_state <= FETCH_BUFFER_STATE;
			RUN_OP:
			    next_state <= COMPUTE_STATE;
			LOAD_OP:
			    next_state <= LOAD_STATE;
			HALT_OP:
			    next_state <= HALT_STATE;
			NOP:
			    next_state <= FETCH_FIFO_STATE;
		    endcase
		FETCH_ADDRESS_STATE:
		    if (buffer_done)
			next_state <= FETCH_FIFO_STATE;	
		FETCH_BUFFER_STATE:
		    if (buffer_done)
			next_state <= FETCH_FIFO_STATE;
		LOAD_STATE:
		    if (buffer_done)
			next_state <= FETCH_FIFO_STATE;
		COMPUTE_STATE: // THIS MIGHT WORK as it just waits for stored final value
		    if (buffer_done)
			next_state <= FETCH_FIFO_STATE;
		STORE_STATE:
		    if (buffer_done)
			next_state <= FETCH_FIFO_STATE;
	    endcase
	end
    end


    always_ff @(posedge clk) begin
	case (current_state)
	    RESET_STATE: begin // THIS IS NOT FINISHED TODO
		instruction_half <= 1'b0;
		buffer_re        <= 1'b0;
		buffer_we        <= 1'b0;
		rx_re            <= 1'b0;
		tx_we            <= 1'b0;
		compute_start    <= 1'b0;
	        fetch_mode       <= FETCH_INSTRUCTION;		
	    end
	    // before you enter, you must set fetch_mode and
	    // instruction_half to 0
	    FETCH_FIFO_STATE: begin
		case (fetch_mode)
		    FETCH_INSTRUCTION: begin
			if (~rx_we) begin
			    if (~rx_empty && ~instruction_half) begin
				rx_re            <= 1'b1;
				instruction[FIFO_DATA_WIDTH-1:0] <= rx_fifo_to_mem;
				rx_re            <= 1'b0;
				instruction_half <= 1'b1;
			    end else if (~rx_empty && instruction_half) begin
				rx_re            <= 1'b1;
				instruction[BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH] <= rx_fifo_to_mem;
				rx_re            <= 1'b0;
				instruction_half <= 1'b0;
			    end
			end
		    end
		    FETCH_ADDRESS: begin
			if (~rx_empty && ~instruction_half) begin
			    rx_re            <= 1'b1;
			    address[FIFO_DATA_WIDTH-1:0] <= rx_fifo_to_mem;
			    rx_re            <= 1'b0;
			    instruction_half <= 1'b1;
			end else if (~rx_empty && instruction_half) begin
			    rx_re            <= 1'b1;
			    address[ADDRESS_SIZE-1:FIFO_DATA_WIDTH] <= rx_fifo_to_mem;
			    rx_re            <= 1'b0;
			    instruction_half <= 1'b0;
			end
		    end
		endcase
	    end
	    DECODE_STATE: begin
		case (opcode)
		    STORE_OP: begin	
			address_indicator <= (instruction[4]) ? 1'b1 : 1'b0;
			fetch_mode        <= FETCH_ADDRESS;
		    end
		    FETCH_OP: begin
			bot_mem    <= (instruction[3]) ? 1'b1 : 1'b0;
			address    <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE-1];
			fetch_mode <= FETCH_INSTRUCTION;
		    end
		    RUN_OP: begin
			compute_en   <= (instruction[3]) ? 1'b1 : 1'b0;
			quantizer_en <= (instruction[4]) ? 1'b1 : 1'b0;
			relu_en      <= (instruction[5]) ? 1'b1 : 1'b0;
			address      <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE-1];
		    end
		    LOAD_OP: begin
			compute_load_en <= (instruction[3]) ? 1'b1 : 1'b0;
			address         <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE-1];
		    end
		    NOP: 
			fetch_mode <= FETCH_INSTRUCTION;
		endcase
	    end
	    FETCH_ADDRESS_STATE: begin
		if (~tx_full) begin
		    compute_load_en <= 1'b0;
		    buffer_re       <= 1'b1;
		    buffer_we       <= 1'b0;
		    buffer_compute_en <= 1'b1;
		    if (buffer_done) begin
			store_val <= mem_to_compute[;
			buffer_re <= 1'b0;
			buffer_compute_en <= 1'b0;
		    end
		end
	    end
	    FETCH_BUFFER_STATE: begin
		buffer_we <= 1'b1;
		buffer_re <= 1'b0;
		buffer_fifo_en <= 1'b1;
		buffer_compute_en <= 1'b0;
		section <= bot_mem;
		if (buffer_done) begin
		    buffer_fifo_en <= 1'b0;
		    buffer_we <= 1'b1;
		end
	    end
	    LOAD_STATE: begin
		compute_en <= 1'b0; // maybe not needed 
		buffer_re <= 1'b1;
		buffer_compute_en <= 1'b1;
		if (buffer_done) begin
		    compute_in <= mem_to_compute;
		    buffer_re <= 1'b0;
		    buffer_compute_en <= 1'b0;
		end
	    end
	    COMPUTE_STATE: begin
		if (~compute_en && ~quantizer_en && relu_en) begin
		    relu_in           <= mem_to_compute;
		    compute_to_buffer <= relu_out; 
		end else if (~compute_en && quantizer_en && relu_en) begin
		    quantizer_in      <= mem_to_compute;
		    relu_in           <= quantizer_out;
		    compute_to_buffer <= relu_out;
		end else begin
		    compute_in        <= mem_to_compute;
		    quantizer_in      <= compute_out;
		    relu_in           <= quantizer_out;
		    compute_to_buffer <= relu_out;
		end
	    end
	    STORE_STATE: begin
		buffer_we <= 1'b1;
		buffer_re <= 1'b0;
		buffer_compute_en <= 1'b1;
		compute_to_buffer <= store_val;
	    end
	end
    end

    // UART communication happens at the same time as everything else
    always_ff @(posedge clk) begin: uart communication
	// Rx Control
	if (~rx_re && rx_valid)
	    rx_we <= 1'b1;
	else
	    rx_we <= 1'b0;

	//Tx Control
	if (~tx_we && ~tx_empty)
	    tx_re <= 1'b1;
	else 
	    tx_re <= 1'b0;

    end



endmodule: top
