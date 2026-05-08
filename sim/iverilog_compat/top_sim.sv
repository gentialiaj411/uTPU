/*
* Top Module - Iverilog Compatible Version
* Replaces array slice assignments with explicit for loops
*/

`timescale 1ns/1ps

module top #(
    parameter UART_BITS_TRANSFERED   = 8,
    parameter UART_INPUT_CLK         = 100000000,
    parameter UART_BAUD              = 115200,
    parameter FORCE_UART_AA          = 0,
    parameter FORCE_UART_ECHO        = 0,
    parameter ALPHA                  = 2,
    parameter COMPUTE_DATA_WIDTH     = 4,
    parameter ACCUMULATOR_DATA_WIDTH = 16, 
    parameter ARRAY_SIZE             = 16,
    parameter ARRAY_SIZE_WIDTH       = $clog2(ARRAY_SIZE),
    parameter FIFO_WIDTH             = 256,
    parameter FIFO_DATA_WIDTH        = 8,
    parameter BUFFER_SIZE            = 512,
    parameter BUFFER_WORD_SIZE       = 16,
    parameter ADDRESS_SIZE           = $clog2(BUFFER_SIZE),
    parameter OPCODE_WIDTH           = 3,
    parameter RELU_SIZE              = ARRAY_SIZE*ARRAY_SIZE,
    parameter RELU_SIZE_WIDTH        = $clog2(RELU_SIZE),
    parameter QUANTIZER_SIZE         = ARRAY_SIZE*ARRAY_SIZE,
    parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE),
    parameter NUM_COMPUTE_LANES      = ARRAY_SIZE*ARRAY_SIZE,
    parameter STORE_DATA_WIDTH       = 16,
    parameter DEBUG_STORE_ACK        = 0,
    parameter DEBUG_FETCH_ACK        = 0
) (
    input  logic clk, rst,
    input  logic rx,
    output logic tx,
    output logic led_rst
);

    // Controller registers
    logic [ADDRESS_SIZE-1:0]     address; 
    logic                        compute_en;
    logic                        quantizer_en;
    logic                        relu_en;
    logic                        bot_mem;
    logic                        mem_section;
    logic [BUFFER_WORD_SIZE-1:0] store_val;    
    logic [ADDRESS_SIZE-1:0]     store_src_addr;
    logic [ADDRESS_SIZE-1:0]     store_dest_addr;
    logic [ADDRESS_SIZE-1:0]     compute_result_addr;
    logic                        store_ready;
    logic                        store_half;
    logic [1:0]                  store_word_idx;
    logic [7:0]                  store_byte_lo;
    logic [BUFFER_WORD_SIZE-1:0] store_word2;
    logic [BUFFER_WORD_SIZE-1:0] store_word3;
    
    // FIFO receiver signals
    logic rx_we, rx_re, rx_empty, rx_full, rx_valid;
    logic rx_we_d;
    logic [FIFO_DATA_WIDTH-1:0] rx_data_buf;
    logic rx_rvalid, rx_pending;
    logic [FIFO_DATA_WIDTH-1:0] rx_to_fifo;
    logic [FIFO_DATA_WIDTH-1:0] rx_fifo_to_mem;

    // FIFO transmitter signals
    logic tx_we, tx_re, tx_empty, tx_full, tx_start;
    logic [FIFO_DATA_WIDTH-1:0] tx_to_fifo;
    logic [FIFO_DATA_WIDTH-1:0] mem_to_tx_fifo;
    logic [FIFO_DATA_WIDTH-1:0] tx_wdata;
    logic                       tx_pending;
    logic [FIFO_DATA_WIDTH-1:0] tx_pending_data;
    logic                       tx_busy;
    logic                       tx_pop_inflight;
    logic                       tx_start_mux;
    logic [UART_BITS_TRANSFERED-1:0] tx_message_mux;

    // MAC Array signals
    logic compute_start, compute_load_en, compute_done;
    logic signed [COMPUTE_DATA_WIDTH-1:0]     compute_in  [NUM_COMPUTE_LANES-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0];

    // Quantizer signals
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] quantizer_in  [QUANTIZER_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     quantizer_out [QUANTIZER_SIZE-1:0];

    // ReLU signals (packed for iverilog compatibility)
    logic [RELU_SIZE*COMPUTE_DATA_WIDTH-1:0] relu_in_packed;
    logic [RELU_SIZE*COMPUTE_DATA_WIDTH-1:0] relu_out_packed;
    // Internal unpacked arrays for convenience
    logic signed [COMPUTE_DATA_WIDTH-1:0] relu_in  [RELU_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] relu_out [RELU_SIZE-1:0];

    // Buffer signals
    logic buffer_we, buffer_re, buffer_compute_en, buffer_fifo_en, buffer_done, section, buffer_store_en;
    // IVERILOG FIX: Use packed vectors for buffer ports, then unpack internally
    logic [NUM_COMPUTE_LANES*COMPUTE_DATA_WIDTH-1:0] mem_to_compute_packed;
    logic [NUM_COMPUTE_LANES*COMPUTE_DATA_WIDTH-1:0] compute_to_buffer_packed;
    // Internal unpacked arrays for convenience
    logic signed [COMPUTE_DATA_WIDTH-1:0] mem_to_compute    [NUM_COMPUTE_LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] compute_to_buffer [NUM_COMPUTE_LANES-1:0];
    logic [STORE_DATA_WIDTH-1:0]   controller_to_buffer;
    logic [STORE_DATA_WIDTH-1:0]   buffer_to_controller;
    logic [23:0] rst_blink;
    logic        rx_led;
    logic        rst_int;
    logic [23:0] uart_spam_div;
    logic        tx_echo_pending;
    logic [7:0]  tx_echo_data;
    logic        store_ack_pending;
    logic [7:0]  store_ack_data;
    logic [7:0]  debug_tx_q [0:7];
    logic [3:0]  debug_tx_q_count;
    logic [3:0]  debug_tx_q_rd;
    logic [7:0]  rx_trace [0:5];
    logic        buffer_done_d;

    assign rst_int = ~rst;

    uart #(
        .UART_BITS_TRANSFERED(UART_BITS_TRANSFERED),
        .INPUT_CLK(UART_INPUT_CLK),
        .UART_CLK(UART_BAUD)
    ) u_uart (
        .clk(clk),
        .rst(rst_int),
        .tx_start(tx_start_mux),
        .rx(rx),
        .rx_valid(rx_valid),
        .tx(tx),
        .tx_message(tx_message_mux),
        .rx_result(rx_to_fifo),
        .tx_busy(tx_busy)
    );

    assign tx_start_mux   = tx_start;
    assign tx_message_mux = tx_to_fifo;

    fifo_rx #(
        .FIFO_WIDTH(FIFO_WIDTH),
        .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_in (
        .clk(clk),
        .rst(rst_int),
        .we(rx_we),
        .re(rx_re),
        .valid(rx_valid),
        .empty(rx_empty),
        .full(rx_full),
        .w_data(rx_data_buf),
        .r_data(rx_fifo_to_mem),
        .r_valid(rx_rvalid)
    );

    fifo_tx #(
        .FIFO_WIDTH(FIFO_WIDTH),
        .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_out (
        .clk(clk),
        .rst(rst_int),
        .we(tx_we),
        .re(tx_re),
        .start(tx_start),
        .empty(tx_empty),
        .full(tx_full),
        .w_data(tx_wdata),
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
        .rst(rst_int),
        .compute(compute_start),
        .load_en(compute_load_en),
        .done(compute_done),
        .datas_arr(compute_in),
        .weights_in(compute_in),
        .results_arr(compute_out)
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

    // Pack/unpack ReLU arrays for iverilog compatibility
    genvar gi_relu;
    generate
        for (gi_relu = 0; gi_relu < RELU_SIZE; gi_relu++) begin: gen_relu_pack
            assign relu_in_packed[gi_relu*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH] = relu_in[gi_relu];
            assign relu_out[gi_relu] = relu_out_packed[gi_relu*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH];
        end
    endgenerate

    leaky_relu_array #(
        .RELU_SIZE(RELU_SIZE),
        .RELU_SIZE_WIDTH(RELU_SIZE_WIDTH),
        .ALPHA(ALPHA),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
    ) u_leaky_relu_array (
        .in_packed(relu_in_packed),
        .result_packed(relu_out_packed)
    );

    // Pack/unpack compute arrays for buffer port compatibility
    genvar gi_pack;
    generate
        for (gi_pack = 0; gi_pack < NUM_COMPUTE_LANES; gi_pack++) begin: gen_pack_io
            assign mem_to_compute[gi_pack] = mem_to_compute_packed[gi_pack*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH];
            assign compute_to_buffer_packed[gi_pack*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH] = compute_to_buffer[gi_pack];
        end
    endgenerate

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
        .store_en(buffer_store_en),
        .done(buffer_done),
        .section(section),
        .address(address),
        .fifo_in(rx_fifo_to_mem),
        .fifo_out(mem_to_tx_fifo),
        .compute_in_packed(compute_to_buffer_packed),
        .compute_out_packed(mem_to_compute_packed),
        .store_in(controller_to_buffer),
        .store_out(buffer_to_controller)
    );

    typedef enum logic [3:0] {
        RESET_STATE,
        FETCH_FIFO_STATE,
        DECODE_STATE,
        FETCH_ADDRESS_STATE,
        FETCH_BUFFER_STATE,
        LOAD_STATE,
        COMPUTE_STATE,
        COMPUTE_PIPELINE_STATE,   // Stage 1: quantizer processes
        COMPUTE_PIPELINE2_STATE,  // Stage 2: relu processes (if needed)
        COMPUTE_WRITEBACK_STATE,
        STORE_STATE,
        HALT_STATE
    } state_e;

    state_e current_state;
    state_e next_state;

    typedef enum logic [OPCODE_WIDTH-1:0] {
        STORE_OP = 3'b000,
        FETCH_OP = 3'b001,
        RUN_OP   = 3'b010,
        LOAD_OP  = 3'b011,
        HALT_OP  = 3'b100,
        NOP      = 3'b101
    } opcode_e;
    
    typedef enum logic {
        FETCH_INSTRUCTION,
        FETCH_ADDRESS
    } fetch_mode_e;

    logic [BUFFER_WORD_SIZE-1:0] instruction;
    logic                        instruction_half;
    logic                        fetch_bot;
    logic                        instr_ready;

    opcode_e opcode;
    assign opcode = opcode_e'(instruction[OPCODE_WIDTH-1:0]);

    fetch_mode_e fetch_mode;
    logic address_indicator;

    // State register
    always_ff @(posedge clk) begin
        if (rst_int)
            current_state <= RESET_STATE;
        else 
            current_state <= next_state;
    end

    // Buffer done delay
    always_ff @(posedge clk) begin
        if (rst_int)
            buffer_done_d <= 1'b0;
        else
            buffer_done_d <= buffer_done;
    end

    // Next state logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            RESET_STATE:
                next_state = FETCH_FIFO_STATE;
            FETCH_FIFO_STATE: begin
                if (fetch_mode == FETCH_INSTRUCTION) begin
                    if (instr_ready)
                        next_state = DECODE_STATE;
                end else if (fetch_mode == FETCH_ADDRESS && store_ready) begin
                    if (address_indicator)
                        next_state = STORE_STATE;
                    else
                        next_state = FETCH_ADDRESS_STATE;
                end
            end
            DECODE_STATE:
                case (opcode)
                    STORE_OP: next_state = FETCH_FIFO_STATE;
                    FETCH_OP: next_state = FETCH_BUFFER_STATE;
                    RUN_OP:   next_state = COMPUTE_STATE;
                    LOAD_OP:  next_state = LOAD_STATE;
                    HALT_OP:  next_state = HALT_STATE;
                    NOP:      next_state = FETCH_FIFO_STATE;
                    default:  next_state = FETCH_FIFO_STATE;
                endcase
            FETCH_ADDRESS_STATE:
                if (buffer_done_d) next_state = STORE_STATE;
            FETCH_BUFFER_STATE:
                if (buffer_done_d) next_state = FETCH_FIFO_STATE;
            LOAD_STATE:
                if (buffer_done_d) next_state = FETCH_FIFO_STATE;
            COMPUTE_STATE:
                if (compute_done) next_state = COMPUTE_PIPELINE_STATE;
            COMPUTE_PIPELINE_STATE:
                next_state = COMPUTE_PIPELINE2_STATE;
            COMPUTE_PIPELINE2_STATE:
                next_state = COMPUTE_WRITEBACK_STATE;
            COMPUTE_WRITEBACK_STATE:
                if (buffer_done) next_state = FETCH_FIFO_STATE;
            STORE_STATE:
                if (buffer_done) next_state = FETCH_FIFO_STATE;
            default: next_state = RESET_STATE;
        endcase
    end

    logic tx_selftest_sent;

    // Main FSM output logic
    always_ff @(posedge clk) begin
        tx_we <= 1'b0;
        case (current_state)
            RESET_STATE: begin
                instruction_half <= 1'b0;
                instr_ready      <= 1'b0;
                rx_pending       <= 1'b0;
                buffer_re        <= 1'b0;
                buffer_we        <= 1'b0;
                rx_re            <= 1'b0;
                compute_start    <= 1'b0;
                fetch_mode       <= FETCH_INSTRUCTION;
                fetch_bot        <= '0;
                tx_pending       <= 1'b0;
                tx_pending_data  <= '0;
                tx_selftest_sent <= 1'b0;
                uart_spam_div    <= '0;
                tx_echo_pending  <= 1'b0;
                tx_echo_data     <= '0;
                rx_led           <= 1'b0;
                store_src_addr      <= '0;
                store_dest_addr     <= '0;
                compute_result_addr <= '0;
                store_half          <= 1'b0;
                store_word_idx   <= 2'b0;
                store_byte_lo    <= '0;
                store_word2      <= '0;
                store_word3      <= '0;
                store_ack_pending<= 1'b0;
                store_ack_data   <= 8'h00;
                store_ready      <= 1'b0;
                debug_tx_q_count <= 4'd0;
                debug_tx_q_rd    <= 4'd0;
                for (int di = 0; di < 8; di++) debug_tx_q[di] <= 8'h00;
                for (int dj = 0; dj < 6; dj++) rx_trace[dj] <= 8'h00;
                compute_en       <= 1'b0;
                quantizer_en     <= 1'b0;
                relu_en          <= 1'b0;
                compute_load_en  <= 1'b0;
                buffer_fifo_en   <= 1'b0;
                buffer_compute_en<= 1'b0;
                buffer_store_en  <= 1'b0;
            end

            FETCH_FIFO_STATE: begin
                rx_re <= 1'b0;
                if (rx_rvalid) begin
                    rx_pending <= 1'b0;
                    case (fetch_mode)
                        FETCH_INSTRUCTION: begin
                            if (~instruction_half) begin
                                instruction[FIFO_DATA_WIDTH-1:0] <= rx_fifo_to_mem;
                                instruction_half <= 1'b1;
                                instr_ready <= 1'b0;
                            end else begin
                                instruction[BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH] <= rx_fifo_to_mem;
                                instruction_half <= 1'b0;
                                instr_ready <= 1'b1;
                            end
                        end
                        FETCH_ADDRESS: begin
                            if (~store_half) begin
                                store_byte_lo <= rx_fifo_to_mem;
                                store_half <= 1'b1;
                            end else begin
                                store_half <= 1'b0;
                                if (store_word_idx == 2'b0) begin
                                    store_word2 <= {rx_fifo_to_mem, store_byte_lo};
                                    store_word_idx <= 2'b1;
                                    if (!address_indicator)
                                        store_src_addr <= {rx_fifo_to_mem, store_byte_lo};
                                end else begin
                                    store_word3 <= {rx_fifo_to_mem, store_byte_lo};
                                    store_word_idx <= 2'b0;
                                    store_dest_addr <= {rx_fifo_to_mem, store_byte_lo};
                                    if (address_indicator)
                                        store_val <= store_word2;
                                    store_ready <= 1'b1;
                                end
                            end
                        end
                    endcase
                end else if (~rx_pending && ~rx_we && ~rx_empty) begin
                    rx_re <= 1'b1;
                    rx_pending <= 1'b1;
                end
            end

            DECODE_STATE: begin
                rx_re <= '0;
                instr_ready <= 1'b0;
                case (opcode)
                    STORE_OP: begin
                        address_indicator <= (instruction[4]) ? 1'b1 : 1'b0;
                        fetch_mode <= FETCH_ADDRESS;
                        instruction_half <= 1'b0;
                        store_half <= 1'b0;
                        store_word_idx <= 2'b0;
                    end
                    FETCH_OP: begin
                        bot_mem    <= (instruction[3]) ? 1'b1 : 1'b0;
                        address    <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                        fetch_mode <= FETCH_INSTRUCTION;
                    end
                    RUN_OP: begin
                        compute_en   <= (instruction[3]) ? 1'b1 : 1'b0;
                        quantizer_en <= (instruction[4]) ? 1'b1 : 1'b0;
                        relu_en      <= (instruction[5]) ? 1'b1 : 1'b0;
                        address      <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                        compute_result_addr <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    LOAD_OP: begin
                        compute_load_en <= (instruction[3]) ? 1'b1 : 1'b0;
                        address         <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    NOP:
                        fetch_mode <= FETCH_INSTRUCTION;
                    default: ;
                endcase
            end

            FETCH_ADDRESS_STATE: begin
                rx_re <= '0;
                if (~tx_full) begin
                    compute_load_en <= 1'b0;
                    address         <= store_src_addr;
                    buffer_re       <= 1'b1;
                    buffer_we       <= 1'b0;
                    buffer_store_en <= 1'b1;
                    if (buffer_done_d) begin
                        store_val       <= buffer_to_controller[BUFFER_WORD_SIZE-1:0];
                        address         <= store_dest_addr;
                        buffer_re       <= 1'b0;
                        buffer_store_en <= 1'b0;
                    end
                end
            end

            FETCH_BUFFER_STATE: begin
                buffer_we         <= 1'b0;
                buffer_re         <= 1'b1;
                buffer_fifo_en    <= 1'b1;
                buffer_compute_en <= 1'b0;
                section           <= bot_mem;
                if (buffer_done_d) begin
                    buffer_fifo_en   <= 1'b0;
                    buffer_re        <= 1'b0;
                    tx_pending       <= 1'b1;
                    instruction_half <= 1'b0;  // FIX: Reset for next instruction
                    fetch_mode       <= FETCH_INSTRUCTION;  // FIX: Ensure correct mode
                    if (DEBUG_FETCH_ACK)
                        tx_pending_data <= 8'hCC;
                    else
                        tx_pending_data <= mem_to_tx_fifo;
                end
            end

            LOAD_STATE: begin
                compute_en        <= 1'b0;
                buffer_re         <= 1'b1;
                buffer_compute_en <= 1'b1;
                // Debug: show what we see
                $display("[FSM LOAD] buffer_done=%0d, buffer_done_d=%0d, mem_to_compute[0:3]=%0d,%0d,%0d,%0d",
                    buffer_done, buffer_done_d, 
                    $signed(mem_to_compute[0]), $signed(mem_to_compute[1]),
                    $signed(mem_to_compute[2]), $signed(mem_to_compute[3]));
                if (buffer_done_d) begin
                    $display("[FSM LOAD] Copying to compute_in!");
                    // IVERILOG COMPATIBLE: explicit loop instead of array slice assignment
                    for (int i = 0; i < NUM_COMPUTE_LANES; i++) begin
                        compute_in[i] <= mem_to_compute[i];
                    end
                    buffer_re         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                end
            end

            COMPUTE_STATE: begin
                compute_start <= 1'b1;
                
                // When compute is done, set up the first stage of output pipeline
                if (compute_done) begin
                    compute_start <= '0;
                    
                    // Stage 1: Feed inputs to quantizer (if needed) or relu (if only relu)
                    if (~compute_en && ~quantizer_en && relu_en) begin
                        // ReLU only - feed pre-loaded data to relu
                        for (int i = 0; i < RELU_SIZE; i++) begin
                            relu_in[i] <= compute_in[i];
                        end
                    end else if (compute_en) begin
                        // Compute was used - feed results to quantizer
                        for (int i = 0; i < QUANTIZER_SIZE; i++) begin
                            quantizer_in[i] <= compute_out[i];
                        end
                    end
                end
            end
            
            COMPUTE_PIPELINE_STATE: begin
                // Stage 2: quantizer_out is now valid (if used)
                if (~compute_en && ~quantizer_en && relu_en) begin
                    // ReLU only - relu_out is now valid, capture it
                    for (int i = 0; i < NUM_COMPUTE_LANES; i++) begin
                        compute_to_buffer[i] <= relu_out[i];
                    end
                end else if (compute_en && quantizer_en && ~relu_en) begin
                    // Compute + Quantizer only - quantizer_out is now valid
                    for (int i = 0; i < NUM_COMPUTE_LANES; i++) begin
                        compute_to_buffer[i] <= quantizer_out[i];
                    end
                end else if (compute_en && quantizer_en && relu_en) begin
                    // Compute + Quantizer + ReLU - feed quantizer_out to relu
                    for (int i = 0; i < RELU_SIZE; i++) begin
                        relu_in[i] <= quantizer_out[i];
                    end
                end else if (compute_en && ~quantizer_en && ~relu_en) begin
                    // Compute only - truncate to 4 bits
                    for (int i = 0; i < NUM_COMPUTE_LANES; i++) begin
                        compute_to_buffer[i] <= compute_out[i][COMPUTE_DATA_WIDTH-1:0];
                    end
                end
            end
            
            COMPUTE_PIPELINE2_STATE: begin
                // Stage 3: relu_out is now valid (for full pipeline)
                if (compute_en && quantizer_en && relu_en) begin
                    // Compute + Quantizer + ReLU - relu_out is now valid
                    for (int i = 0; i < NUM_COMPUTE_LANES; i++) begin
                        compute_to_buffer[i] <= relu_out[i];
                    end
                end
                // Other modes already captured in COMPUTE_PIPELINE_STATE
            end

            COMPUTE_WRITEBACK_STATE: begin
                // Write compute_to_buffer back to unified buffer at result address
                buffer_we         <= 1'b1;
                buffer_re         <= 1'b0;
                buffer_fifo_en    <= 1'b0;
                buffer_compute_en <= 1'b1;  // Use compute interface for parallel write
                buffer_store_en   <= 1'b0;
                address           <= compute_result_addr;
                if (buffer_done) begin
                    buffer_we         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                    fetch_mode        <= FETCH_INSTRUCTION;
                    instruction_half  <= 1'b0;
                end
            end

            STORE_STATE: begin
                buffer_we         <= 1'b1;
                buffer_re         <= 1'b0;
                buffer_fifo_en    <= 1'b0;
                buffer_compute_en <= 1'b0;
                buffer_store_en   <= 1'b1;
                address           <= store_dest_addr;
                controller_to_buffer <= store_val;
                if (buffer_done) begin
                    buffer_we       <= '0;
                    buffer_store_en <= '0;
                    fetch_mode      <= FETCH_INSTRUCTION;
                    instruction_half <= 1'b0;
                    store_ready     <= 1'b0;
                    store_half      <= 1'b0;
                    store_word_idx  <= 2'b0;
                    store_ack_pending <= 1'b0;
                end
            end

            default: ;
        endcase

        // TX path
        if (FORCE_UART_AA) begin
            uart_spam_div <= uart_spam_div + 1'b1;
            if (~tx_full && uart_spam_div == '0) begin
                tx_we <= 1'b1;
                tx_wdata <= 8'hAA;
            end
        end else if (FORCE_UART_ECHO) begin
            if (rx_valid && ~tx_full) begin
                tx_we <= 1'b1;
                tx_wdata <= rx_to_fifo;
            end
        end else if (DEBUG_STORE_ACK && store_ack_pending && ~tx_full) begin
            tx_we <= 1'b1;
            tx_wdata <= store_ack_data;
            store_ack_pending <= 1'b0;
        end else if (DEBUG_STORE_ACK && debug_tx_q_count != 0 && ~tx_full) begin
            tx_we <= 1'b1;
            tx_wdata <= debug_tx_q[debug_tx_q_rd];
            debug_tx_q_rd <= debug_tx_q_rd + 1'b1;
            debug_tx_q_count <= debug_tx_q_count - 1'b1;
        end else begin
            if (~tx_selftest_sent && ~tx_full) begin
                tx_we <= 1'b1;
                tx_wdata <= 8'hAA;
                tx_selftest_sent <= 1'b1;
            end

            if (tx_pending && ~tx_full) begin
                tx_we    <= 1'b1;
                tx_wdata <= tx_pending_data;
                tx_pending <= 1'b0;
            end
        end
    end

    // UART control
    always_ff @(posedge clk) begin
        if (rst_int) begin
            rx_we_d <= 1'b0;
            tx_re <= 1'b0;
            tx_pop_inflight <= 1'b0;
        end else begin
            if (rx_valid && ~rx_full) begin
                rx_data_buf <= rx_to_fifo;
                rx_we_d     <= 1'b1;
                rx_led      <= ~rx_led;
                rx_trace[5] <= rx_trace[4];
                rx_trace[4] <= rx_trace[3];
                rx_trace[3] <= rx_trace[2];
                rx_trace[2] <= rx_trace[1];
                rx_trace[1] <= rx_trace[0];
                rx_trace[0] <= rx_to_fifo;
            end else begin
                rx_we_d <= 1'b0;
            end

            tx_re <= 1'b0;
            if (!tx_pop_inflight && ~tx_empty && ~tx_busy && ~tx_we) begin
                tx_re <= 1'b1;
                tx_pop_inflight <= 1'b1;
            end else if (tx_busy) begin
                tx_pop_inflight <= 1'b0;
            end
        end
    end

    assign rx_we = rx_we_d;
    assign led_rst = (~rst_int & rst_blink[23]) ^ rx_led;

    always_ff @(posedge clk) begin
        if (~rst_int)
            rst_blink <= rst_blink + 1'b1;
        else
            rst_blink <= '0;
    end

endmodule: top
