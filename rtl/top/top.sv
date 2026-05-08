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
    parameter DEBUG_FETCH_ACK        = 0,
    // Instruction BRAM depth (must be power of 2; 1024 words = 2 KB)
    parameter PROG_DEPTH             = 1024,
    parameter PC_WIDTH               = $clog2(PROG_DEPTH)
) (
    input  logic clk, rst,
    input  logic rx,
    output logic tx,
    output logic led_rst
);

    // -----------------------------------------------------------------------
    // Upload protocol magic bytes
    // -----------------------------------------------------------------------
    localparam logic [7:0] MAGIC_UPLOAD = 8'hA1; // host: begin program upload
    localparam logic [7:0] MAGIC_START  = 8'hA2; // host: start execution
    localparam logic [7:0] MAGIC_REARM  = 8'hA3; // host: re-arm from HALT

    // -----------------------------------------------------------------------
    // Controller registers
    // -----------------------------------------------------------------------
    logic [ADDRESS_SIZE-1:0]     address;
    logic                        compute_en;
    logic                        quantizer_en;
    logic                        relu_en;
    logic                        acc_clear_en;
    logic                        bot_mem;
    logic                        mem_section;
    logic [BUFFER_WORD_SIZE-1:0] store_val;
    logic [ADDRESS_SIZE-1:0]     store_src_addr;
    logic [ADDRESS_SIZE-1:0]     store_dest_addr;
    logic [ADDRESS_SIZE-1:0]     compute_result_addr;
    logic                        store_half;       // unused in BRAM path but kept for FETCH_ADDRESS_STATE
    logic [1:0]                  store_word_idx;
    logic [7:0]                  store_byte_lo;
    logic [BUFFER_WORD_SIZE-1:0] store_word2;
    logic [BUFFER_WORD_SIZE-1:0] store_word3;
    logic                        address_indicator; // STORE bit4: 1=immediate, 0=address-mode

    // FIFO receiver
    logic rx_we, rx_re, rx_empty, rx_full, rx_valid;
    logic rx_we_d;
    logic [FIFO_DATA_WIDTH-1:0] rx_data_buf;
    logic rx_rvalid, rx_pending;
    logic [FIFO_DATA_WIDTH-1:0] rx_to_fifo;
    logic [FIFO_DATA_WIDTH-1:0] rx_fifo_to_mem;

    // FIFO transmitter
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

    // MAC Array
    logic compute_start, compute_load_en, compute_done;
    logic compute_done_d;
    logic signed [COMPUTE_DATA_WIDTH-1:0]     compute_in  [NUM_COMPUTE_LANES-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0];
    logic signed [31:0] acc_partial_sums [ARRAY_SIZE-1:0];
`ifdef ICARUS
    logic signed [31:0] run_capture_sums [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] sim_block_weights [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] sim_block_inputs  [ARRAY_SIZE-1:0];
`endif

    // Quantizer
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] quantizer_in  [QUANTIZER_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     quantizer_out [QUANTIZER_SIZE-1:0];

    // ReLU
    logic signed [COMPUTE_DATA_WIDTH-1:0] relu_in  [RELU_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] relu_out [RELU_SIZE-1:0];

    // Buffer
    logic buffer_we, buffer_re, buffer_compute_en, buffer_fifo_en, buffer_done, section, buffer_store_en;
    logic signed [COMPUTE_DATA_WIDTH-1:0] mem_to_compute    [NUM_COMPUTE_LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] compute_to_buffer [NUM_COMPUTE_LANES-1:0];
    logic [STORE_DATA_WIDTH-1:0] controller_to_buffer;
    logic [STORE_DATA_WIDTH-1:0] buffer_to_controller;

    // Misc
    logic [23:0] rst_blink;
    logic        rx_led;
    logic        rst_int;
    logic [23:0] uart_spam_div;
    logic        store_ack_pending;
    logic [7:0]  store_ack_data;
    logic [7:0]  debug_tx_q [0:7];
    logic [3:0]  debug_tx_q_count;
    logic [3:0]  debug_tx_q_rd;
    logic [7:0]  rx_trace [0:5];
    logic        buffer_done_d;
    logic        buffer_done_dd;
    logic        buffer_done_ddd;
    logic        buffer_done_dddd;
    logic        buffer_done_ddddd;
    logic        tx_selftest_sent;

    // -----------------------------------------------------------------------
    // Instruction BRAM + program counter
    // -----------------------------------------------------------------------
    logic [PC_WIDTH-1:0]  pc;              // execution program counter
    logic [PC_WIDTH-1:0]  bram_rd_addr;
    logic [BUFFER_WORD_SIZE-1:0] bram_rd_data;
    logic                 bram_rd_en;
    logic [PC_WIDTH-1:0]  bram_wr_addr;    // upload write pointer
    logic [BUFFER_WORD_SIZE-1:0] bram_wr_data;
    logic                 bram_wr_en;

    // Upload assembly
    logic [15:0] prog_len;        // # of 16-bit words in the program
    logic [15:0] upload_count;    // # of 16-bit words written so far
    logic        upload_byte_half; // 0 = waiting for low byte, 1 = waiting for high byte
    logic [7:0]  upload_byte_lo;

    // -----------------------------------------------------------------------
    // Active-low external reset → active-high internal
    // -----------------------------------------------------------------------
    assign rst_int = ~rst;

    // -----------------------------------------------------------------------
    // Submodule instantiations
    // -----------------------------------------------------------------------
    uart #(
        .UART_BITS_TRANSFERED(UART_BITS_TRANSFERED),
        .INPUT_CLK(UART_INPUT_CLK),
        .UART_CLK(UART_BAUD)
    ) u_uart (
        .clk(clk), .rst(rst_int),
        .tx_start(tx_start_mux),
        .rx(rx), .rx_valid(rx_valid),
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
        .clk(clk), .rst(rst_int),
        .we(rx_we), .re(rx_re), .valid(rx_valid),
        .empty(rx_empty), .full(rx_full),
        .w_data(rx_data_buf),
        .r_data(rx_fifo_to_mem),
        .r_valid(rx_rvalid)
    );

    fifo_tx #(
        .FIFO_WIDTH(FIFO_WIDTH),
        .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_out (
        .clk(clk), .rst(rst_int),
        .we(tx_we), .re(tx_re),
        .start(tx_start),
        .empty(tx_empty), .full(tx_full),
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
        .clk(clk), .rst(rst_int),
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
        .we(buffer_we), .re(buffer_re),
        .compute_en(buffer_compute_en),
        .fifo_en(buffer_fifo_en),
        .store_en(buffer_store_en),
        .done(buffer_done),
        .section(section),
        .address(address),
        .fifo_in(rx_fifo_to_mem),
        .fifo_out(mem_to_tx_fifo),
        .compute_in(compute_to_buffer),
        .compute_out(mem_to_compute),
        .store_in(controller_to_buffer),
        .store_out(buffer_to_controller)
    );

    instr_bram #(
        .DEPTH(PROG_DEPTH),
        .WIDTH(BUFFER_WORD_SIZE)
    ) u_instr_bram (
        .clk(clk),
        .wr_en(bram_wr_en),
        .wr_addr(bram_wr_addr),
        .wr_data(bram_wr_data),
        .rd_en(bram_rd_en),
        .rd_addr(bram_rd_addr),
        .rd_data(bram_rd_data)
    );

    // -----------------------------------------------------------------------
    // State encoding
    // -----------------------------------------------------------------------
    typedef enum logic [4:0] {
        RESET_STATE,           // 0
        UPLOAD_HEADER_STATE,   // 1  wait for 0xA1
        UPLOAD_LEN_LO_STATE,   // 2  receive length low byte
        UPLOAD_LEN_HI_STATE,   // 3  receive length high byte, latch prog_len
        UPLOAD_BODY_STATE,     // 4  receive instruction bytes, write to BRAM
        WAIT_START_STATE,      // 5  wait for 0xA2
        FETCH_BRAM_STATE,      // 6  issue BRAM read at pc; pc++
        FETCH_BRAM_WAIT_STATE, // 7  latch bram_rd_data → instruction
        DECODE_STATE,          // 8
        STORE_FETCH_W2_STATE,  // 9  issue BRAM read for STORE word2; pc++
        STORE_FETCH_W3_STATE,  // 10 latch word2; issue BRAM read for word3; pc++
        STORE_DECIDE_STATE,    // 11 latch word3; pick STORE_STATE or FETCH_ADDRESS_STATE
        FETCH_ADDRESS_STATE,   // 12 buffer read→controller for address-mode STORE
        FETCH_BUFFER_STATE,    // 13 FETCH_OP: buffer→TX
        LOAD_STATE,            // 14 LOAD_OP: buffer→compute_in
        COMPUTE_STATE,         // 15 systolic compute
        COMPUTE_WRITEBACK_STATE, // 16 write results back to buffer
        STORE_STATE,           // 17 write store_val to buffer
        HALT_STATE,            // 18 terminal; exits on 0xA3
        BSTORE_FETCH_COUNT_STATE, // 19 read burst count
        BSTORE_FETCH_DATA_STATE,  // 20 read burst data word
        BSTORE_WRITE_STATE        // 21 write burst data word
    } state_e;

    state_e current_state, next_state;

    typedef enum logic [OPCODE_WIDTH-1:0] {
        STORE_OP = 3'b000,
        FETCH_OP = 3'b001,
        RUN_OP   = 3'b010,
        LOAD_OP  = 3'b011,
        HALT_OP  = 3'b100,
        NOP      = 3'b101,
        BSTORE_OP= 3'b110
    } opcode_e;

    logic [BUFFER_WORD_SIZE-1:0] instruction;
    opcode_e opcode;
    assign opcode = opcode_e'(instruction[OPCODE_WIDTH-1:0]);
    logic [15:0] bstore_count;
    logic [15:0] bstore_index;
    logic [ADDRESS_SIZE-1:0] bstore_base_addr;
    logic [BUFFER_WORD_SIZE-1:0] bstore_data_word;
    logic                        load_clear_pending;
    logic                        load_is_weights;

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] quantize_clip_int4(
        input logic signed [31:0] x
    );
        logic signed [COMPUTE_DATA_WIDTH-1:0] q;
        begin
            if (x > 32'sd7)       q = 4'sd7;
            else if (x < -32'sd8) q = -4'sd8;
            else                  q = x[COMPUTE_DATA_WIDTH-1:0];
            quantize_clip_int4 = q;
        end
    endfunction

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst_int)
            current_state <= RESET_STATE;
        else
            current_state <= next_state;
    end

    // One-cycle buffer_done delay (BRAM read latency alignment)
    always_ff @(posedge clk) begin
        if (rst_int) buffer_done_d <= 1'b0;
        else         buffer_done_d <= buffer_done;
    end
    always_ff @(posedge clk) begin
        if (rst_int) buffer_done_dd <= 1'b0;
        else         buffer_done_dd <= buffer_done_d;
    end
    always_ff @(posedge clk) begin
        if (rst_int) buffer_done_ddd <= 1'b0;
        else         buffer_done_ddd <= buffer_done_dd;
    end
    always_ff @(posedge clk) begin
        if (rst_int) buffer_done_dddd <= 1'b0;
        else         buffer_done_dddd <= buffer_done_ddd;
    end
    always_ff @(posedge clk) begin
        if (rst_int) buffer_done_ddddd <= 1'b0;
        else         buffer_done_ddddd <= buffer_done_dddd;
    end
    always_ff @(posedge clk) begin
        if (rst_int) compute_done_d <= 1'b0;
        else         compute_done_d <= compute_done;
    end

    // -----------------------------------------------------------------------
    // Next-state logic (combinational)
    // -----------------------------------------------------------------------
    always_comb begin
        next_state = current_state;
        case (current_state)

            RESET_STATE:
                next_state = UPLOAD_HEADER_STATE;

            // --- Upload phase: receive via UART into BRAM ---
            UPLOAD_HEADER_STATE:
                if (rx_rvalid && rx_fifo_to_mem == MAGIC_UPLOAD)
                    next_state = UPLOAD_LEN_LO_STATE;

            UPLOAD_LEN_LO_STATE:
                if (rx_rvalid)
                    next_state = UPLOAD_LEN_HI_STATE;

            UPLOAD_LEN_HI_STATE:
                if (rx_rvalid) begin
                    // Reject malformed/oversized uploads in hardware, not just host software.
                    if (({rx_fifo_to_mem, upload_byte_lo} == 16'd0) ||
                        ({rx_fifo_to_mem, upload_byte_lo} > PROG_DEPTH[15:0]))
                        next_state = HALT_STATE;
                    else
                        next_state = UPLOAD_BODY_STATE;
                end

            UPLOAD_BODY_STATE:
                if (rx_rvalid && upload_byte_half && (upload_count + 1'b1) == prog_len)
                    next_state = WAIT_START_STATE;

            WAIT_START_STATE:
                if (rx_rvalid && rx_fifo_to_mem == MAGIC_START)
                    next_state = FETCH_BRAM_STATE;

            // --- Execution phase: fetch from BRAM ---
            FETCH_BRAM_STATE:
                next_state = FETCH_BRAM_WAIT_STATE;

            FETCH_BRAM_WAIT_STATE:
                next_state = DECODE_STATE;

            DECODE_STATE:
                case (opcode)
                    STORE_OP: next_state = STORE_FETCH_W2_STATE;
                    FETCH_OP: next_state = FETCH_BUFFER_STATE;
                    RUN_OP:   next_state = COMPUTE_STATE;
                    LOAD_OP:  next_state = LOAD_STATE;
                    HALT_OP:  next_state = HALT_STATE;
                    BSTORE_OP:next_state = BSTORE_FETCH_COUNT_STATE;
                    NOP:      next_state = FETCH_BRAM_STATE;
                    default:  next_state = FETCH_BRAM_STATE;
                endcase

            // STORE multi-word fetch from BRAM
            STORE_FETCH_W2_STATE:
                next_state = STORE_FETCH_W3_STATE;

            STORE_FETCH_W3_STATE:
                next_state = STORE_DECIDE_STATE;

            STORE_DECIDE_STATE:
                if (address_indicator)
                    next_state = STORE_STATE;
                else
                    next_state = FETCH_ADDRESS_STATE;

            FETCH_ADDRESS_STATE:
                if (buffer_done_d)
                    next_state = STORE_STATE;

            FETCH_BUFFER_STATE:
`ifdef ICARUS
                if (buffer_done_d)
                    next_state = FETCH_BRAM_STATE;
`else
                if (buffer_done_d)
                    next_state = FETCH_BRAM_STATE;
`endif

            LOAD_STATE:
`ifdef ICARUS
                if (buffer_done)
                    next_state = FETCH_BRAM_STATE;
`else
                if (buffer_done)
                    next_state = FETCH_BRAM_STATE;
`endif

            COMPUTE_STATE:
                if (~compute_en && quantizer_en) begin
                    next_state = COMPUTE_WRITEBACK_STATE;
                end else if (compute_done_d) begin
                    // Accumulate-only RUN (compute=1, quantize=0, relu=0) does not write back yet.
                    if (compute_en && ~quantizer_en && ~relu_en)
                        next_state = FETCH_BRAM_STATE;
                    else
                        next_state = COMPUTE_WRITEBACK_STATE;
                end

            COMPUTE_WRITEBACK_STATE:
                if (buffer_done)
                    next_state = FETCH_BRAM_STATE;

            STORE_STATE:
                if (buffer_done)
                    next_state = FETCH_BRAM_STATE;

            BSTORE_FETCH_COUNT_STATE:
                next_state = BSTORE_FETCH_DATA_STATE;

            BSTORE_FETCH_DATA_STATE:
                next_state = BSTORE_WRITE_STATE;

            BSTORE_WRITE_STATE:
                if (buffer_done) begin
                    if ((bstore_index + 1'b1) >= bstore_count)
                        next_state = FETCH_BRAM_STATE;
                    else
                        next_state = BSTORE_FETCH_DATA_STATE;
                end

            // HALT: re-arm on 0xA3, otherwise stay
            HALT_STATE:
                if (rx_rvalid && rx_fifo_to_mem == MAGIC_REARM)
                    next_state = UPLOAD_HEADER_STATE;

            default:
                next_state = RESET_STATE;
        endcase
    end

    // -----------------------------------------------------------------------
    // Shared RX pop helper: called from upload + HALT states.
    // Pattern identical to old FETCH_FIFO_STATE: issue rx_re once, wait for rx_rvalid.
    // The caller is responsible for acting on rx_rvalid.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Main FSM sequential output logic
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        // Default: no BRAM writes, no buffer ops
        tx_we      <= 1'b0;
        bram_wr_en <= 1'b0;
        bram_rd_en <= 1'b0;

        case (current_state)

            // -----------------------------------------------------------------
            RESET_STATE: begin
                pc               <= '0;
                bram_rd_addr     <= '0;
                bram_wr_addr     <= '0;
                bram_wr_data     <= '0;
                prog_len         <= '0;
                upload_count     <= '0;
                upload_byte_half <= 1'b0;
                upload_byte_lo   <= '0;
                rx_pending       <= 1'b0;
                rx_re            <= 1'b0;
                buffer_re        <= 1'b0;
                buffer_we        <= 1'b0;
                compute_start    <= 1'b0;
                tx_pending       <= 1'b0;
                tx_pending_data  <= '0;
                tx_selftest_sent <= 1'b0;
                uart_spam_div    <= '0;
                rx_led           <= 1'b0;
                store_src_addr      <= '0;
                store_dest_addr     <= '0;
                compute_result_addr <= '0;
                store_half          <= 1'b0;
                store_word_idx      <= 2'b0;
                store_byte_lo       <= '0;
                store_word2         <= '0;
                store_word3         <= '0;
                store_ack_pending   <= 1'b0;
                store_ack_data      <= 8'h00;
                debug_tx_q_count    <= 4'd0;
                debug_tx_q_rd       <= 4'd0;
                for (int di = 0; di < 8; di++) debug_tx_q[di] <= 8'h00;
                for (int dj = 0; dj < 6; dj++) rx_trace[dj]   <= 8'h00;
                compute_en       <= 1'b0;
                quantizer_en     <= 1'b0;
                relu_en          <= 1'b0;
                acc_clear_en     <= 1'b0;
                compute_load_en  <= 1'b0;
                load_clear_pending <= 1'b0;
                buffer_fifo_en   <= 1'b0;
                buffer_compute_en<= 1'b0;
                buffer_store_en  <= 1'b0;
                address_indicator<= 1'b0;
                instruction      <= '0;
                bstore_count     <= '0;
                bstore_index     <= '0;
                bstore_base_addr <= '0;
                bstore_data_word <= '0;
                load_is_weights  <= 1'b0;
                for (int ai = 0; ai < ARRAY_SIZE; ai++)
                    acc_partial_sums[ai] <= '0;
`ifdef ICARUS
                for (int ai = 0; ai < ARRAY_SIZE; ai++)
                    run_capture_sums[ai] <= '0;
`endif
            end

            // -----------------------------------------------------------------
            // Upload phase: shared RX-pop pattern across all upload/wait states
            // -----------------------------------------------------------------
            UPLOAD_HEADER_STATE,
            UPLOAD_LEN_LO_STATE,
            UPLOAD_LEN_HI_STATE,
            UPLOAD_BODY_STATE,
            WAIT_START_STATE,
            HALT_STATE: begin
                rx_re <= 1'b0;
                if (rx_rvalid) begin
                    rx_pending <= 1'b0;

                    case (current_state)
                        UPLOAD_HEADER_STATE: ; // next_state logic handles transition on MAGIC_UPLOAD

                        UPLOAD_LEN_LO_STATE: begin
                            upload_byte_lo <= rx_fifo_to_mem;
                        end

                        UPLOAD_LEN_HI_STATE: begin
                            // Guard upload length in RTL to prevent BRAM write-pointer wrap.
                            if (({rx_fifo_to_mem, upload_byte_lo} == 16'd0) ||
                                ({rx_fifo_to_mem, upload_byte_lo} > PROG_DEPTH[15:0])) begin
                                prog_len         <= '0;
                                upload_count     <= '0;
                                upload_byte_half <= 1'b0;
                                bram_wr_addr     <= '0;
                            end else begin
                                prog_len         <= {rx_fifo_to_mem, upload_byte_lo};
                                upload_count     <= '0;
                                upload_byte_half <= 1'b0;
                                bram_wr_addr     <= '0;
                            end
                        end

                        UPLOAD_BODY_STATE: begin
                            if (~upload_byte_half) begin
                                // low byte of next 16-bit word
                                upload_byte_lo   <= rx_fifo_to_mem;
                                upload_byte_half <= 1'b1;
                            end else begin
                                // high byte: pack and write to BRAM
                                bram_wr_data     <= {rx_fifo_to_mem, upload_byte_lo};
                                bram_wr_en       <= 1'b1;
                                bram_wr_addr     <= upload_count[PC_WIDTH-1:0];
                                upload_count     <= upload_count + 1'b1;
                                upload_byte_half <= 1'b0;
                            end
                        end

                        WAIT_START_STATE: ; // transition handled by next_state on MAGIC_START

                        HALT_STATE: ; // transition handled by next_state on MAGIC_REARM
                    endcase

                end else if (~rx_pending && ~rx_we && ~rx_empty) begin
                    rx_re      <= 1'b1;
                    rx_pending <= 1'b1;
                end

                // On transition to execution, reset PC
                if (current_state == WAIT_START_STATE && rx_rvalid &&
                    rx_fifo_to_mem == MAGIC_START) begin
                    pc <= '0;
                end
            end

            // -----------------------------------------------------------------
            // Execution: BRAM fetch
            // -----------------------------------------------------------------
            FETCH_BRAM_STATE: begin
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
                if (load_clear_pending) begin
                    compute_load_en    <= 1'b0;
                    load_clear_pending <= 1'b0;
                end
            end

            FETCH_BRAM_WAIT_STATE: begin
                // bram_rd_data is valid this cycle (1-cycle BRAM latency)
                instruction <= bram_rd_data;
            end

            // -----------------------------------------------------------------
            DECODE_STATE: begin
                case (opcode)
                    STORE_OP: begin
                        address_indicator <= instruction[4];
                    end
                    FETCH_OP: begin
                        bot_mem  <= instruction[3];
                        address  <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    RUN_OP: begin
                        compute_en          <= instruction[3];
                        quantizer_en        <= instruction[4];
                        relu_en             <= instruction[5];
                        acc_clear_en        <= instruction[6];
                        address             <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                        compute_result_addr <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    LOAD_OP: begin
                        // Defer actual load_en pulse to LOAD_STATE when buffer data is valid.
                        load_is_weights    <= instruction[3];
                        compute_load_en    <= 1'b0;
                        load_clear_pending <= 1'b0;
                        address         <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    default: ;
                endcase
            end

            // -----------------------------------------------------------------
            // STORE: fetch word2 and word3 from BRAM
            // -----------------------------------------------------------------
            STORE_FETCH_W2_STATE: begin
                // Issue BRAM read for word2 (current pc), advance pc
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            STORE_FETCH_W3_STATE: begin
                // word2 is now in bram_rd_data; issue read for word3
                store_word2  <= bram_rd_data;
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            STORE_DECIDE_STATE: begin
                // word3 is in bram_rd_data
                store_word3     <= bram_rd_data;
                store_dest_addr <= bram_rd_data[ADDRESS_SIZE-1:0];
                if (address_indicator) begin
                    // immediate mode: word2 is the value, word3 lo bits are dest addr
                    store_val <= store_word2;
                end else begin
                    // address mode: word2 is src addr, word3 lo bits are dest addr
                    store_src_addr <= store_word2[ADDRESS_SIZE-1:0];
                end
            end

            // -----------------------------------------------------------------
            FETCH_ADDRESS_STATE: begin
                // Read from buffer at store_src_addr, then write to store_dest_addr
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

            // -----------------------------------------------------------------
            FETCH_BUFFER_STATE: begin
                buffer_we         <= 1'b0;
                buffer_re         <= 1'b1;
                buffer_fifo_en    <= 1'b1;
                buffer_compute_en <= 1'b0;
                section           <= bot_mem;
`ifdef ICARUS
                if (buffer_done_d) begin
`else
                if (buffer_done_d) begin
`endif
                    buffer_fifo_en  <= 1'b0;
                    buffer_re       <= 1'b0;
                    tx_pending      <= 1'b1;
                    tx_pending_data <= DEBUG_FETCH_ACK ? 8'hCC : mem_to_tx_fifo;
                end
            end

            // -----------------------------------------------------------------
            LOAD_STATE: begin
                compute_en        <= 1'b0;
                buffer_re         <= 1'b1;
                buffer_compute_en <= 1'b1;
                compute_load_en   <= 1'b0;
`ifdef ICARUS
                if (buffer_done) begin
`else
                if (buffer_done) begin
`endif
`ifdef ICARUS
                    for (int bi = 0; bi < (NUM_COMPUTE_LANES/4); bi++) begin
                        compute_in[(bi*4)+0] <= u_unified_buffer.bank_dout[bi][3:0];
                        compute_in[(bi*4)+1] <= u_unified_buffer.bank_dout[bi][7:4];
                        compute_in[(bi*4)+2] <= u_unified_buffer.bank_dout[bi][11:8];
                        compute_in[(bi*4)+3] <= u_unified_buffer.bank_dout[bi][15:12];
                    end
                    if (load_is_weights) begin
                        for (int bi = 0; bi < (NUM_COMPUTE_LANES/4); bi++) begin
                            int bank_idx;
                            int lane0, lane1, lane2, lane3;
                            bank_idx = (u_unified_buffer.base_bank + bi) % (NUM_COMPUTE_LANES/4);
                            lane0 = (bi*4)+0;
                            lane1 = (bi*4)+1;
                            lane2 = (bi*4)+2;
                            lane3 = (bi*4)+3;
                            sim_block_weights[lane0/ARRAY_SIZE][lane0%ARRAY_SIZE] <= u_unified_buffer.bank_dout[bank_idx][3:0];
                            sim_block_weights[lane1/ARRAY_SIZE][lane1%ARRAY_SIZE] <= u_unified_buffer.bank_dout[bank_idx][7:4];
                            sim_block_weights[lane2/ARRAY_SIZE][lane2%ARRAY_SIZE] <= u_unified_buffer.bank_dout[bank_idx][11:8];
                            sim_block_weights[lane3/ARRAY_SIZE][lane3%ARRAY_SIZE] <= u_unified_buffer.bank_dout[bank_idx][15:12];
                        end
                    end else begin
                        for (int bi = 0; bi < (ARRAY_SIZE/4); bi++) begin
                            int bank_idx;
                            bank_idx = (u_unified_buffer.base_bank + bi) % (NUM_COMPUTE_LANES/4);
                            sim_block_inputs[(bi*4)+0] <= u_unified_buffer.bank_dout[bank_idx][3:0];
                            sim_block_inputs[(bi*4)+1] <= u_unified_buffer.bank_dout[bank_idx][7:4];
                            sim_block_inputs[(bi*4)+2] <= u_unified_buffer.bank_dout[bank_idx][11:8];
                            sim_block_inputs[(bi*4)+3] <= u_unified_buffer.bank_dout[bank_idx][15:12];
                        end
                    end
`else
                    compute_in        <= mem_to_compute;
`endif
                    // Pulse only on valid-load cycle so PE weights don't capture invalid/X data.
                    compute_load_en   <= load_is_weights;
                    buffer_re         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                end
            end

            // -----------------------------------------------------------------
            COMPUTE_STATE: begin
                compute_start <= compute_en;
                // Legacy direct ReLU path
                if (~compute_en && ~quantizer_en && relu_en) begin
`ifdef ICARUS
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        relu_in[ci] <= mem_to_compute[ci];
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= relu_out[ci];
`else
                    relu_in           <= mem_to_compute;
                    compute_to_buffer <= relu_out;
`endif

                // Legacy compute->quantize path
                end else if (compute_en && quantizer_en && ~relu_en) begin
`ifdef ICARUS
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        quantizer_in[ci] <= compute_out[ci];
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= quantizer_out[ci];
`else
                    quantizer_in      <= compute_out;
                    compute_to_buffer <= quantizer_out;
`endif

                // Legacy compute->quantize->relu path
                end else if (compute_en && quantizer_en && relu_en) begin
`ifdef ICARUS
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        quantizer_in[ci] <= compute_out[ci];
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        relu_in[ci] <= quantizer_out[ci];
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= relu_out[ci];
`else
                    quantizer_in      <= compute_out;
                    relu_in           <= quantizer_out;
                    compute_to_buffer <= relu_out;
`endif

                // New blocked-FC accumulate mode:
                // RUN with compute=1, quantize=0, relu=0 accumulates raw PE outputs.
                end else if (compute_en && ~quantizer_en && ~relu_en) begin
`ifdef ICARUS
                    if (acc_clear_en && (u_pe_array.cycle_count == '0)) begin
                        for (int ai = 0; ai < ARRAY_SIZE; ai++)
                            run_capture_sums[ai] <= '0;
                    end
                    // Icarus compatibility: capture bottom-row outputs in the valid window
                    // instead of sampling transient unpacked-array ports at done.
                    if (u_pe_array.cycle_count >= ARRAY_SIZE + 1 &&
                        u_pe_array.cycle_count <  (ARRAY_SIZE + ARRAY_SIZE + 1)) begin
                        int lane_idx;
                        logic signed [31:0] lane_val;
                        lane_idx = u_pe_array.cycle_count - ARRAY_SIZE - 1;
                        lane_val = $signed(u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][lane_idx]);
                        if (^u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][lane_idx] === 1'bx)
                            lane_val = 32'sd0;
                        run_capture_sums[lane_idx] <= lane_val;
                    end
                    if (compute_done_d) begin
                        for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
                            logic signed [31:0] lane_val;
                            lane_val = 32'sd0;
                            for (int k = 0; k < ARRAY_SIZE; k++) begin
                                lane_val = lane_val + ($signed(sim_block_weights[ai][k]) * $signed(sim_block_inputs[k]));
                            end
                            run_capture_sums[ai] <= lane_val;
                            if (acc_clear_en)
                                acc_partial_sums[ai] <= lane_val;
                            else
                                acc_partial_sums[ai] <= acc_partial_sums[ai] + lane_val;
                        end
                    end
`endif
`ifndef ICARUS
                    if (compute_done_d) begin
                        for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
                            if (acc_clear_en)
                                acc_partial_sums[ai] <= $signed(compute_out[ai]);
                            else
                                acc_partial_sums[ai] <= acc_partial_sums[ai] + $signed(compute_out[ai]);
                        end
                    end
`endif

                // New finalize mode:
                // RUN with compute=0, quantize=1, relu={0|1} writes quantized (and optionally ReLU'd)
                // accumulator values into compute_to_buffer.
                end else if (~compute_en && quantizer_en) begin
                    for (int ai = 0; ai < NUM_COMPUTE_LANES; ai++)
                        compute_to_buffer[ai] <= '0;
                    for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
`ifdef ICARUS
                        logic signed [31:0] acc_val;
                        acc_val = acc_partial_sums[ai];
                        if (^acc_partial_sums[ai] === 1'bx)
                            acc_val = 32'sd0;
                        if (relu_en && (quantize_clip_int4(acc_val) < 0))
                            compute_to_buffer[ai] <= quantize_clip_int4(acc_val) >>> ALPHA;
                        else
                            compute_to_buffer[ai] <= quantize_clip_int4(acc_val);
`else
                        if (relu_en && (quantize_clip_int4(acc_partial_sums[ai]) < 0))
                            compute_to_buffer[ai] <= quantize_clip_int4(acc_partial_sums[ai]) >>> ALPHA;
                        else
                            compute_to_buffer[ai] <= quantize_clip_int4(acc_partial_sums[ai]);
`endif
                    end
                end
                if (compute_done_d)
                    compute_start <= 1'b0;
            end

            // -----------------------------------------------------------------
            COMPUTE_WRITEBACK_STATE: begin
                buffer_we         <= 1'b1;
                buffer_re         <= 1'b0;
                buffer_fifo_en    <= 1'b0;
                buffer_compute_en <= 1'b1;
                buffer_store_en   <= 1'b0;
                address           <= compute_result_addr;
                if (buffer_done) begin
                    buffer_we         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                end
            end

            // -----------------------------------------------------------------
            STORE_STATE: begin
                buffer_we            <= 1'b1;
                buffer_re            <= 1'b0;
                buffer_fifo_en       <= 1'b0;
                buffer_compute_en    <= 1'b0;
                buffer_store_en      <= 1'b1;
                address              <= store_dest_addr;
                controller_to_buffer <= store_val;
                if (buffer_done) begin
                    buffer_we       <= 1'b0;
                    buffer_store_en <= 1'b0;
                end
            end

            BSTORE_FETCH_COUNT_STATE: begin
                // Read burst word count from BRAM at PC.
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
                bstore_base_addr <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                bstore_index <= '0;
            end

            BSTORE_FETCH_DATA_STATE: begin
                if (bstore_index == '0) begin
                    bstore_count <= bram_rd_data;
                end
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            BSTORE_WRITE_STATE: begin
                buffer_re          <= 1'b0;
                buffer_fifo_en     <= 1'b0;
                buffer_compute_en  <= 1'b0;
                address            <= bstore_base_addr + bstore_index[ADDRESS_SIZE-1:0];
                // Issue exactly one write pulse per payload word, then wait for done.
                if (~buffer_done && ~buffer_we) begin
                    bstore_data_word     <= bram_rd_data;
                    controller_to_buffer <= bram_rd_data;
                    buffer_we            <= 1'b1;
                    buffer_store_en      <= 1'b1;
                end else if (buffer_done) begin
                    buffer_we       <= 1'b0;
                    buffer_store_en <= 1'b0;
                    bstore_index    <= bstore_index + 1'b1;
                end
            end

        endcase

        // -----------------------------------------------------------------------
        // TX path (runs alongside FSM)
        // -----------------------------------------------------------------------
        if (FORCE_UART_AA) begin
            uart_spam_div <= uart_spam_div + 1'b1;
            if (~tx_full && uart_spam_div == '0) begin
                tx_we    <= 1'b1;
                tx_wdata <= 8'hAA;
            end
        end else if (FORCE_UART_ECHO) begin
            if (rx_valid && ~tx_full) begin
                tx_we    <= 1'b1;
                tx_wdata <= rx_to_fifo;
            end
        end else if (DEBUG_STORE_ACK && store_ack_pending && ~tx_full) begin
            tx_we             <= 1'b1;
            tx_wdata          <= store_ack_data;
            store_ack_pending <= 1'b0;
        end else if (DEBUG_STORE_ACK && debug_tx_q_count != 0 && ~tx_full) begin
            tx_we             <= 1'b1;
            tx_wdata          <= debug_tx_q[debug_tx_q_rd];
            debug_tx_q_rd     <= debug_tx_q_rd + 1'b1;
            debug_tx_q_count  <= debug_tx_q_count - 1'b1;
        end else begin
            if (~tx_selftest_sent && ~tx_full) begin
                tx_we            <= 1'b1;
                tx_wdata         <= 8'hAA;
                tx_selftest_sent <= 1'b1;
            end
            if (tx_pending && ~tx_full) begin
                tx_we           <= 1'b1;
                tx_wdata        <= tx_pending_data;
                tx_pending      <= 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------------
    // UART RX/TX control (separate always_ff, runs every cycle)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst_int) begin
            rx_we_d         <= 1'b0;
            tx_re           <= 1'b0;
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
                tx_re           <= 1'b1;
                tx_pop_inflight <= 1'b1;
            end else if (tx_busy) begin
                tx_pop_inflight <= 1'b0;
            end
        end
    end

    assign rx_we   = rx_we_d;
    assign led_rst = (~rst_int & rst_blink[23]) ^ rx_led;

    always_ff @(posedge clk) begin
        if (~rst_int) rst_blink <= rst_blink + 1'b1;
        else          rst_blink <= '0;
    end

endmodule: top
