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
    // Phase 4 widening: when 1, FETCH/RUN/LOAD/BSTORE use a 2-word extended
    // address layout (low ADDRESS_SIZE bits of the next instruction word).
    // When 0 (default), legacy 9-bit address-in-opcode-word layout is used
    // and behaviour is byte-identical to pre-Phase-4.
    parameter EXT_ADDR_EN            = 0,
    parameter OPCODE_WIDTH           = 3,
    // Finalize datapath width. Default ARRAY_SIZE matches the PE output rate
    // (one column of rows per cycle). Override to ARRAY_SIZE*ARRAY_SIZE to
    // restore the legacy one-cycle tile-wide requant/ReLU (A/B testable).
    parameter QUANTIZER_LANES        = ARRAY_SIZE,
    parameter RELU_LANES             = ARRAY_SIZE,
    // 0=combo (default until Fmax proves pipeline worth it), 1=Step2b,
    // 3=product/shift/clamp stages (DSP MREG target).
    parameter int QUANTIZER_PIPE_DEPTH = 0,
    parameter RELU_SIZE              = RELU_LANES,
    parameter RELU_SIZE_WIDTH        = $clog2(RELU_SIZE),
    parameter QUANTIZER_SIZE         = QUANTIZER_LANES,
    parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE),
    parameter NUM_COMPUTE_LANES      = ARRAY_SIZE*ARRAY_SIZE,
    parameter MAX_BATCH_COUNT        = 64,
    parameter MAX_BATCH_COUNT_WIDTH  = $clog2(MAX_BATCH_COUNT + 1),
    parameter MAX_STREAM_LANES       = ARRAY_SIZE*MAX_BATCH_COUNT,
    // Phase 4 widening: derived elements-per-buffer-word. INT4@16-bit word
    // -> 4; INT8@16-bit word -> 2. Drives the LOAD_STATE pack/unpack loops
    // and matches `unified_buffer.sv`'s `ITEMS_IN_SLOT` calculation.
    parameter ITEMS_IN_SLOT          = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH,
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
    localparam logic [7:0] MAGIC_READ_PERF = 8'hA4; // host: stream perf counters
    localparam logic [3:0] NOP_SUBOP_REQUANT = 4'b0001;
    localparam int STREAM_CHUNK_WORDS = (ARRAY_SIZE * ARRAY_SIZE) / ITEMS_IN_SLOT;
    // Narrow finalize: QUANTIZER_SIZE == ARRAY_SIZE streams one output column
    // per cycle. Wide finalize: QUANTIZER_SIZE == NUM_COMPUTE_LANES keeps the
    // legacy one-cycle tile requant.
    localparam bit REQUANT_NARROW = (QUANTIZER_SIZE == ARRAY_SIZE);

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
    logic [MAX_BATCH_COUNT_WIDTH-1:0] input_batch_count;
    logic [MAX_BATCH_COUNT_WIDTH-1:0] run_batch_count;
    logic [MAX_BATCH_COUNT_WIDTH-1:0] run_batch_index;
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
    logic compute_weight_commit;
    logic compute_done_d;
    logic signed [COMPUTE_DATA_WIDTH-1:0]     compute_weights_in [NUM_COMPUTE_LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     compute_stream_in  [MAX_STREAM_LANES-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] compute_stream_out [MAX_STREAM_LANES-1:0];
    logic signed [31:0] acc_partial_sums [ARRAY_SIZE-1:0];
    logic signed [31:0] acc_partial_matrix [ARRAY_SIZE-1:0][MAX_BATCH_COUNT-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] loaded_input_matrix [ARRAY_SIZE-1:0][MAX_BATCH_COUNT-1:0];
`ifdef ICARUS
    logic signed [31:0] run_capture_sums [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] sim_block_weights [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] sim_block_inputs  [ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] sim_block_inputs_matrix [ARRAY_SIZE-1:0][MAX_BATCH_COUNT-1:0];
`endif

    // Quantizer
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] quantizer_in  [QUANTIZER_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     quantizer_out [QUANTIZER_SIZE-1:0];
    logic [(QUANTIZER_SIZE*ACCUMULATOR_DATA_WIDTH)-1:0] quantizer_ins_flat;
    logic [(QUANTIZER_SIZE*COMPUTE_DATA_WIDTH)-1:0] quantizer_out_flat;
    logic [(QUANTIZER_SIZE*16)-1:0]           requant_multiplier_flat;
    logic [(QUANTIZER_SIZE*16)-1:0]           requant_right_shift_flat;

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
    logic [63:0] perf_cycle_counter;
    logic [63:0] perf_busy_counter;
    logic [63:0] perf_program_count;
    logic [191:0] perf_snapshot;
    logic         perf_stream_active;
    logic [4:0]   perf_stream_idx;
    logic         perf_busy_active;
    logic         perf_waiting_for_immediate_finalize;
    logic         perf_span_measuring;
    logic [63:0]  perf_compute_span_counter;

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
        .NUM_COMPUTE_LANES(NUM_COMPUTE_LANES),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .BATCH_COUNT_WIDTH(MAX_BATCH_COUNT_WIDTH)
    ) u_pe_array (
        .clk(clk), .rst(rst_int),
        .compute(compute_start),
        .load_en(compute_load_en),
        .weight_commit(compute_weight_commit),
        .batch_count(run_batch_count),
        .done(compute_done),
        .datas_arr(compute_stream_in),
        .weights_in(compute_weights_in),
        .results_arr(compute_stream_out)
    );

    quantizer_array #(
        .QUANTIZER_SIZE(QUANTIZER_SIZE),
        .QUANTIZER_SIZE_WIDTH(QUANTIZER_SIZE_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .QUANTIZER_PIPE_DEPTH(QUANTIZER_PIPE_DEPTH)
    ) u_quantizer_array (
        .clk(clk),
        .rst(rst_int),
        .ins_flat(quantizer_ins_flat),
        .requant_enable(requant_finalize_enable),
        .requant_multiplier_flat(requant_multiplier_flat),
        .requant_right_shift_flat(requant_right_shift_flat),
        .results_flat(quantizer_out_flat)
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

    generate
        for (genvar qi = 0; qi < QUANTIZER_SIZE; qi++) begin: quantizer_bus_map
            localparam int QI_IN_LO = qi * ACCUMULATOR_DATA_WIDTH;
            localparam int QI_OUT_LO = qi * COMPUTE_DATA_WIDTH;
            localparam int QI_PARAM_LO = qi * 16;
            localparam int QI_ROW = qi % ARRAY_SIZE;
            assign quantizer_ins_flat[QI_IN_LO +: ACCUMULATOR_DATA_WIDTH] = quantizer_in[qi];
            assign quantizer_out[qi] = quantizer_out_flat[QI_OUT_LO +: COMPUTE_DATA_WIDTH];
            assign requant_multiplier_flat[QI_PARAM_LO +: 16] = requant_multiplier_latched[QI_ROW];
            assign requant_right_shift_flat[QI_PARAM_LO +: 16] = requant_right_shift_latched[QI_ROW];
        end
    endgenerate

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
        RESIDUAL_FETCH_STATE,   // 15 residual buffer fetch for widened RUN path
        COMPUTE_STATE,         // 16 systolic compute
        COMPUTE_WRITEBACK_STATE, // 17 write results back to buffer
        STORE_STATE,           // 18 write store_val to buffer
        HALT_STATE,            // 19 terminal; exits on 0xA3
        BSTORE_FETCH_COUNT_STATE, // 20 read burst count
        BSTORE_FETCH_DATA_STATE,  // 21 read burst data word
        BSTORE_WRITE_STATE,       // 22 write burst data word
        EXT_ADDR_FETCH_STATE,     // 23 (EXT_ADDR_EN=1) issue BRAM read for address word; pc++
        EXT_ADDR_LATCH_STATE,     // 24 (EXT_ADDR_EN=1) latch address word, route to target state
        REQUANT_FETCH_MULT_STATE, // 25 read requant multiplier payload
        REQUANT_FETCH_SHIFT_STATE,// 26 latch multiplier, read right-shift payload
        REQUANT_LATCH_STATE       // 27 latch right-shift payload
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
    logic                        residual_en;
    logic [ADDRESS_SIZE-1:0]     residual_source_addr;
    logic                        run_residual_addr_stage;
    logic signed [COMPUTE_DATA_WIDTH-1:0] residual_in [NUM_COMPUTE_LANES-1:0];
    logic [MAX_BATCH_COUNT_WIDTH-1:0] load_chunk_index;
    logic [MAX_BATCH_COUNT_WIDTH-1:0] load_chunk_count;
    logic                             load_wait_clear;
    logic [MAX_BATCH_COUNT_WIDTH-1:0] writeback_chunk_index;
    logic [MAX_BATCH_COUNT_WIDTH-1:0] writeback_chunk_count;
    logic                             writeback_wait_clear;
    // Column counters for narrow (QUANTIZER_SIZE==ARRAY_SIZE) finalize streaming.
    // Width must hold ARRAY_SIZE inclusive (cols_in_chunk in 1..ARRAY_SIZE).
    logic [$clog2(ARRAY_SIZE+1)-1:0]  writeback_col_index;
    logic [$clog2(ARRAY_SIZE+1)-1:0]  writeback_cols_in_chunk;
    // Pipeline fill remaining before capturing registered quantizer outputs.
    // QUANTIZER_PIPE_DEPTH==0 (combo) keeps this at 0 and captures immediately.
    logic [3:0]                       writeback_pipe_fill_cnt;
    logic                        requant_enable_latched;
    logic [15:0]                 requant_multiplier_latched [ARRAY_SIZE-1:0];
    logic [15:0]                 requant_right_shift_latched [ARRAY_SIZE-1:0];
    logic                        requant_vector_mode_latched;
    logic [7:0]                  requant_vector_count_latched;
    logic [7:0]                  requant_vector_index_latched;
    logic                        requant_finalize_enable;
    assign requant_finalize_enable = requant_enable_latched && !compute_en && quantizer_en;
    // Phase 4: latched opcode for routing the EXT_ADDR_LATCH_STATE -> target
    // state in 2-word extended-address mode. Only used when EXT_ADDR_EN=1.
    opcode_e                     pending_opcode;

    // Phase 4: parameterised saturating clip from 32-bit signed accumulator
    // to ``COMPUTE_DATA_WIDTH`` signed output. Reduces to the original INT4
    // ``+7 / -8`` clip when ``COMPUTE_DATA_WIDTH=4``.
    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] quantize_clip(
        input logic signed [31:0] x
    );
        logic signed [31:0] hi;
        logic signed [31:0] lo;
        logic signed [COMPUTE_DATA_WIDTH-1:0] q;
        begin
            hi = (32'sd1 <<< (COMPUTE_DATA_WIDTH-1)) - 32'sd1;
            lo = -(32'sd1 <<< (COMPUTE_DATA_WIDTH-1));
            if (x > hi)      q = hi[COMPUTE_DATA_WIDTH-1:0];
            else if (x < lo) q = lo[COMPUTE_DATA_WIDTH-1:0];
            else             q = x[COMPUTE_DATA_WIDTH-1:0];
            quantize_clip = q;
        end
    endfunction

    // Legacy alias kept for any external reference; identical to
    // ``quantize_clip`` at ``COMPUTE_DATA_WIDTH=4``.
    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] quantize_clip_int4(
        input logic signed [31:0] x
    );
        quantize_clip_int4 = quantize_clip(x);
    endfunction

    function automatic logic signed [COMPUTE_DATA_WIDTH-1:0] apply_leaky_relu(
        input logic signed [COMPUTE_DATA_WIDTH-1:0] x
    );
        if (x < 0)
            apply_leaky_relu = x >>> ALPHA;
        else
            apply_leaky_relu = x;
    endfunction

    task automatic prepare_writeback_chunk(input int chunk_idx);
        int local_col;
        int global_col;
        int row;
        int out_idx;
        logic signed [31:0] acc_val;
        begin
            // Wide-only helper (QUANTIZER_SIZE == NUM_COMPUTE_LANES). Guard writes
            // so narrow elaboration (QUANTIZER_SIZE == ARRAY_SIZE) stays in-bounds.
            for (int ai = 0; ai < QUANTIZER_SIZE; ai++) begin
                if (requant_enable_latched)
                    quantizer_in[ai] <= '0;
            end
            for (int ai = 0; ai < NUM_COMPUTE_LANES; ai++) begin
                if (!requant_enable_latched)
                    compute_to_buffer[ai] <= '0;
            end
            for (local_col = 0; local_col < ARRAY_SIZE; local_col++) begin
                global_col = (chunk_idx * ARRAY_SIZE) + local_col;
                if (global_col < run_batch_count) begin
                    for (row = 0; row < ARRAY_SIZE; row++) begin
                        out_idx = (local_col * ARRAY_SIZE) + row;
                        acc_val = acc_partial_matrix[row][global_col];
                        if (requant_enable_latched) begin
                            if (out_idx < QUANTIZER_SIZE)
                                quantizer_in[out_idx] <= acc_val[ACCUMULATOR_DATA_WIDTH-1:0];
                        end else if (relu_en && (quantize_clip_int4(acc_val) < 0)) begin
                            compute_to_buffer[out_idx] <= quantize_clip_int4(acc_val) >>> ALPHA;
                        end else begin
                            compute_to_buffer[out_idx] <= quantize_clip_int4(acc_val);
                        end
                    end
                end
            end
        end
    endtask

    // Narrow finalize: present one output column (ARRAY_SIZE rows) to the
    // QUANTIZER_SIZE==ARRAY_SIZE requant array. Per-channel scales map via
    // QI_ROW = qi (row index).
    task automatic prepare_writeback_column(input int chunk_idx, input int local_col);
        int global_col;
        int row;
        logic signed [31:0] acc_val;
        logic signed [31:0] residual_val;
        begin
            for (int ai = 0; ai < QUANTIZER_SIZE; ai++)
                quantizer_in[ai] <= '0;
            if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                for (row = 0; row < ARRAY_SIZE && row < QUANTIZER_SIZE; row++) begin
`ifdef ICARUS
                    acc_val = acc_partial_sums[row];
                    if (^acc_partial_sums[row] === 1'bx)
                        acc_val = 32'sd0;
                    residual_val = residual_en ? $signed(residual_in[row]) : 32'sd0;
                    if (residual_en && (^residual_in[row] === 1'bx))
                        residual_val = 32'sd0;
`else
                    acc_val = acc_partial_sums[row];
                    residual_val = residual_en ? $signed(residual_in[row]) : 32'sd0;
`endif
                    acc_val = acc_val + residual_val;
                    quantizer_in[row] <= acc_val[ACCUMULATOR_DATA_WIDTH-1:0];
                end
            end else begin
                global_col = (chunk_idx * ARRAY_SIZE) + local_col;
                if (global_col < run_batch_count) begin
                    for (row = 0; row < ARRAY_SIZE && row < QUANTIZER_SIZE; row++) begin
                        acc_val = acc_partial_matrix[row][global_col];
                        quantizer_in[row] <= acc_val[ACCUMULATOR_DATA_WIDTH-1:0];
                    end
                end
            end
        end
    endtask

    task automatic capture_writeback_column(input int local_col);
        int row;
        int out_idx;
        begin
            for (row = 0; row < ARRAY_SIZE && row < QUANTIZER_SIZE; row++) begin
                out_idx = (local_col * ARRAY_SIZE) + row;
                if (relu_en)
                    compute_to_buffer[out_idx] <= apply_leaky_relu(quantizer_out[row]);
                else
                    compute_to_buffer[out_idx] <= quantizer_out[row];
            end
        end
    endtask

    task automatic clear_compute_to_buffer();
        begin
            for (int ai = 0; ai < NUM_COMPUTE_LANES; ai++)
                compute_to_buffer[ai] <= '0;
        end
    endtask

    // -----------------------------------------------------------------------
    // State register
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst_int)
            current_state <= RESET_STATE;
        else
            current_state <= next_state;
    end

    always_ff @(posedge clk) begin
        if (rst_int) begin
            perf_cycle_counter <= '0;
            perf_busy_counter <= '0;
            perf_program_count <= '0;
            perf_busy_active <= 1'b0;
            perf_waiting_for_immediate_finalize <= 1'b0;
            perf_span_measuring <= 1'b0;
            perf_compute_span_counter <= '0;
        end else begin
            perf_cycle_counter <= perf_cycle_counter + 1'b1;
            if (perf_busy_active)
                perf_busy_counter <= perf_busy_counter + 1'b1;
            if (perf_span_measuring)
                perf_compute_span_counter <= perf_compute_span_counter + 1'b1;
            if (current_state == DECODE_STATE && opcode == HALT_OP)
                perf_program_count <= perf_program_count + 1'b1;
        end
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
                    FETCH_OP: next_state = EXT_ADDR_EN ? EXT_ADDR_FETCH_STATE : FETCH_BUFFER_STATE;
                    RUN_OP:   next_state = EXT_ADDR_EN ? EXT_ADDR_FETCH_STATE : COMPUTE_STATE;
                    LOAD_OP:  next_state = EXT_ADDR_EN ? EXT_ADDR_FETCH_STATE : LOAD_STATE;
                    HALT_OP:  next_state = HALT_STATE;
                    BSTORE_OP:next_state = EXT_ADDR_EN ? EXT_ADDR_FETCH_STATE : BSTORE_FETCH_COUNT_STATE;
                    NOP: begin
                        if (EXT_ADDR_EN && instruction[3])
                            next_state = REQUANT_FETCH_MULT_STATE;
                        else
                            next_state = FETCH_BRAM_STATE;
                    end
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
                if (buffer_done && !load_wait_clear &&
                    (load_is_weights || ((load_chunk_index + ARRAY_SIZE) >= input_batch_count)))
                    next_state = FETCH_BRAM_STATE;
`else
                if (buffer_done && !load_wait_clear &&
                    (load_is_weights || ((load_chunk_index + ARRAY_SIZE) >= input_batch_count)))
                    next_state = FETCH_BRAM_STATE;
`endif

            RESIDUAL_FETCH_STATE:
`ifdef ICARUS
                if (buffer_done)
                    next_state = COMPUTE_STATE;
`else
                if (buffer_done)
                    next_state = COMPUTE_STATE;
`endif

            COMPUTE_STATE:
                if (~compute_en && quantizer_en) begin
                    next_state = COMPUTE_WRITEBACK_STATE;
                end else if (compute_done_d) begin
                    // Accumulate-only RUN (compute=1, quantize=0, relu=0) does not write back yet.
                    if (compute_en && ~quantizer_en && ~relu_en) begin
                        next_state = FETCH_BRAM_STATE;
                    end
                    else
                        next_state = COMPUTE_WRITEBACK_STATE;
                end

            COMPUTE_WRITEBACK_STATE:
                if (buffer_done && !writeback_wait_clear &&
                    ((writeback_chunk_index + 1'b1) >= writeback_chunk_count))
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

            // Phase 4 EXT_ADDR_EN: 2-word fetch path. Issue BRAM read for the
            // address word at PC, latch on next cycle, then dispatch.
            EXT_ADDR_FETCH_STATE:
                next_state = EXT_ADDR_LATCH_STATE;

            EXT_ADDR_LATCH_STATE:
                case (pending_opcode)
                    FETCH_OP: next_state = FETCH_BUFFER_STATE;
                    RUN_OP: begin
                        if (residual_en && ~run_residual_addr_stage)
                            next_state = EXT_ADDR_FETCH_STATE;
                        else if (residual_en && run_residual_addr_stage)
                            next_state = RESIDUAL_FETCH_STATE;
                        else
                            next_state = COMPUTE_STATE;
                    end
                    LOAD_OP:  next_state = LOAD_STATE;
                    BSTORE_OP:next_state = BSTORE_FETCH_COUNT_STATE;
                    default:  next_state = FETCH_BRAM_STATE;
                endcase

            REQUANT_FETCH_MULT_STATE:
                next_state = REQUANT_FETCH_SHIFT_STATE;

            REQUANT_FETCH_SHIFT_STATE:
                next_state = REQUANT_LATCH_STATE;

            REQUANT_LATCH_STATE:
                if (requant_vector_mode_latched &&
                    ((requant_vector_index_latched + 1'b1) < requant_vector_count_latched))
                    next_state = REQUANT_FETCH_MULT_STATE;
                else
                    next_state = FETCH_BRAM_STATE;

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
                input_batch_count   <= MAX_BATCH_COUNT_WIDTH'(1);
                run_batch_count     <= MAX_BATCH_COUNT_WIDTH'(1);
                run_batch_index     <= '0;
                load_chunk_index    <= '0;
                load_chunk_count    <= MAX_BATCH_COUNT_WIDTH'(1);
                load_wait_clear     <= 1'b0;
                writeback_chunk_index <= '0;
                writeback_chunk_count <= MAX_BATCH_COUNT_WIDTH'(1);
                writeback_wait_clear  <= 1'b0;
                writeback_col_index   <= '0;
                writeback_cols_in_chunk <= ($clog2(ARRAY_SIZE+1))'(1);
                writeback_pipe_fill_cnt <= 4'd0;
                writeback_wait_clear <= 1'b0;
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
                compute_weight_commit <= 1'b0;
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
                residual_en      <= 1'b0;
                residual_source_addr <= '0;
                run_residual_addr_stage <= 1'b0;
                requant_enable_latched <= 1'b0;
                requant_vector_mode_latched <= 1'b0;
                requant_vector_count_latched <= 8'd1;
                requant_vector_index_latched <= 8'd0;
                for (int ri = 0; ri < ARRAY_SIZE; ri++) begin
                    requant_multiplier_latched[ri] <= 16'd1;
                    requant_right_shift_latched[ri] <= 16'd0;
                end
                for (int ai = 0; ai < NUM_COMPUTE_LANES; ai++)
                    residual_in[ai] <= '0;
                for (int qi = 0; qi < QUANTIZER_SIZE; qi++)
                    quantizer_in[qi] <= '0;
                perf_snapshot    <= '0;
                perf_stream_active <= 1'b0;
                perf_stream_idx  <= '0;
                perf_busy_active <= 1'b0;
                perf_waiting_for_immediate_finalize <= 1'b0;
                perf_span_measuring <= 1'b0;
                perf_compute_span_counter <= '0;
                for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
                    acc_partial_sums[ai] <= '0;
                    for (int bj = 0; bj < MAX_BATCH_COUNT; bj++) begin
                        acc_partial_matrix[ai][bj] <= '0;
                        loaded_input_matrix[ai][bj] <= '0;
                    end
                end
`ifdef ICARUS
                for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
                    sim_block_inputs[ai] <= '0;
                    run_capture_sums[ai] <= '0;
                    for (int bj = 0; bj < ARRAY_SIZE; bj++) begin
                        sim_block_weights[ai][bj] <= '0;
                    end
                    for (int bj = 0; bj < MAX_BATCH_COUNT; bj++) begin
                        sim_block_inputs_matrix[ai][bj] <= '0;
                    end
                end
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

                        WAIT_START_STATE: begin
                            if (rx_fifo_to_mem == MAGIC_READ_PERF) begin
                                perf_snapshot <= {perf_cycle_counter, perf_busy_counter, perf_program_count};
                                perf_stream_active <= 1'b1;
                                perf_stream_idx <= '0;
                            end
                        end

                        HALT_STATE: begin
                            if (rx_fifo_to_mem == MAGIC_READ_PERF) begin
                                perf_snapshot <= {perf_cycle_counter, perf_busy_counter, perf_program_count};
                                perf_stream_active <= 1'b1;
                                perf_stream_idx <= '0;
                            end
                        end
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
                if (perf_waiting_for_immediate_finalize) begin
                    if (!(
                        (opcode_e'(bram_rd_data[OPCODE_WIDTH-1:0]) == RUN_OP) &&
                        (bram_rd_data[3] == 1'b0) &&
                        (bram_rd_data[4] == 1'b1)
                    ))
                        perf_busy_active <= 1'b0;
                    perf_waiting_for_immediate_finalize <= 1'b0;
                end
            end

            // -----------------------------------------------------------------
            DECODE_STATE: begin
                pending_opcode <= opcode;
                residual_en <= 1'b0;
                run_residual_addr_stage <= 1'b0;
                case (opcode)
                    STORE_OP: begin
                        address_indicator <= instruction[4];
                    end
                    FETCH_OP: begin
                        bot_mem  <= instruction[3];
                        // Legacy: address embedded in opcode word. Ext-addr:
                        // address comes from the next instruction word and
                        // is latched in EXT_ADDR_LATCH_STATE.
                        if (!EXT_ADDR_EN)
                            address <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    RUN_OP: begin
                        compute_en          <= instruction[3];
                        quantizer_en        <= instruction[4];
                        relu_en             <= instruction[5];
                        acc_clear_en        <= instruction[6];
                        residual_en         <= EXT_ADDR_EN ? instruction[7] : 1'b0;
                        run_batch_count     <= EXT_ADDR_EN ? (instruction[15:8] + MAX_BATCH_COUNT_WIDTH'(1)) : MAX_BATCH_COUNT_WIDTH'(1);
                        run_batch_index     <= '0;
                        run_residual_addr_stage <= 1'b0;
                        if (!EXT_ADDR_EN) begin
                            address             <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                            compute_result_addr <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                        end
                        if (instruction[3]) begin
                            perf_busy_active <= 1'b1;
                            perf_span_measuring <= 1'b1;
                        end else if (instruction[4]) begin
                            perf_busy_active <= 1'b0;
                            perf_span_measuring <= 1'b0;
                        end
                    end
                    LOAD_OP: begin
                        // Defer actual load_en pulse to LOAD_STATE when buffer data is valid.
                        load_is_weights    <= instruction[3];
                        input_batch_count  <= (EXT_ADDR_EN && ~instruction[3]) ? (instruction[15:8] + MAX_BATCH_COUNT_WIDTH'(1)) : MAX_BATCH_COUNT_WIDTH'(1);
                        load_chunk_index   <= '0;
                        load_chunk_count   <= instruction[3]
                            ? MAX_BATCH_COUNT_WIDTH'(1)
                            : ((EXT_ADDR_EN && ~instruction[3] && ((instruction[15:8] + 8'd1) > ARRAY_SIZE))
                                ? (((instruction[15:8] + 8'd1 + ARRAY_SIZE - 1) / ARRAY_SIZE))
                                : MAX_BATCH_COUNT_WIDTH'(1));
                        compute_load_en    <= 1'b0;
                        load_clear_pending <= 1'b0;
                        if (!EXT_ADDR_EN)
                            address <= instruction[BUFFER_WORD_SIZE-1:BUFFER_WORD_SIZE-ADDRESS_SIZE];
                    end
                    NOP: begin
                        if (EXT_ADDR_EN && instruction[3]) begin
                            requant_vector_mode_latched  <= instruction[5];
                            requant_vector_count_latched <= instruction[5] ? (instruction[15:8] + 8'd1) : 8'd1;
                            requant_vector_index_latched <= 8'd0;
                            if (instruction[5]) begin
                                for (int ri = 0; ri < ARRAY_SIZE; ri++) begin
                                    requant_multiplier_latched[ri] <= 16'd1;
                                    requant_right_shift_latched[ri] <= 16'd0;
                                end
                            end
                        end
                    end
                    default: ;
                endcase
            end

            // -----------------------------------------------------------------
            // Phase 4 EXT_ADDR_EN: read the address word from BRAM and latch
            // its low ADDRESS_SIZE bits as the operand address.
            // -----------------------------------------------------------------
            EXT_ADDR_FETCH_STATE: begin
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            EXT_ADDR_LATCH_STATE: begin
                // bram_rd_data holds the address word.
                address <= bram_rd_data[ADDRESS_SIZE-1:0];
                if (pending_opcode == RUN_OP) begin
                    if (residual_en && ~run_residual_addr_stage) begin
                        compute_result_addr <= bram_rd_data[ADDRESS_SIZE-1:0];
                        run_residual_addr_stage <= 1'b1;
                    end else if (residual_en && run_residual_addr_stage) begin
                        residual_source_addr <= bram_rd_data[ADDRESS_SIZE-1:0];
                        run_residual_addr_stage <= 1'b0;
                    end else begin
                        compute_result_addr <= bram_rd_data[ADDRESS_SIZE-1:0];
                    end
                end
                if (pending_opcode == BSTORE_OP)
                    bstore_base_addr <= bram_rd_data[ADDRESS_SIZE-1:0];
            end

            REQUANT_FETCH_MULT_STATE: begin
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            REQUANT_FETCH_SHIFT_STATE: begin
                if (requant_vector_mode_latched)
                    requant_multiplier_latched[requant_vector_index_latched] <= bram_rd_data;
                else begin
                    for (int ri = 0; ri < ARRAY_SIZE; ri++)
                        requant_multiplier_latched[ri] <= bram_rd_data;
                end
                bram_rd_addr <= pc[PC_WIDTH-1:0];
                bram_rd_en   <= 1'b1;
                pc           <= pc + 1'b1;
            end

            REQUANT_LATCH_STATE: begin
                if (requant_vector_mode_latched) begin
                    requant_right_shift_latched[requant_vector_index_latched] <= bram_rd_data;
                    requant_vector_index_latched <= requant_vector_index_latched + 1'b1;
                end else begin
                    for (int ri = 0; ri < ARRAY_SIZE; ri++)
                        requant_right_shift_latched[ri] <= bram_rd_data;
                end
                requant_enable_latched <= instruction[4];
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
                if (buffer_done_d) begin
                    buffer_fifo_en  <= 1'b0;
                    buffer_re       <= 1'b0;
                    tx_pending      <= 1'b1;
                    tx_pending_data <= DEBUG_FETCH_ACK ? 8'hCC : mem_to_tx_fifo;
                end
            end

            // -----------------------------------------------------------------
            LOAD_STATE: begin
                compute_en        <= 1'b0;
                compute_load_en   <= 1'b0;
                if (load_wait_clear) begin
                    buffer_re         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                    if (!buffer_done)
                        load_wait_clear <= 1'b0;
                end else begin
                    buffer_re         <= 1'b1;
                    buffer_compute_en <= 1'b1;
`ifdef ICARUS
                    if (buffer_done) begin
`else
                    if (buffer_done) begin
`endif
`ifdef ICARUS
                    // Phase 4 widening: parameter-driven lane unpack instead
                    // of hardcoded /4 INT4 nibbles. Reduces to the legacy
                    // 4-nibble code path when ITEMS_IN_SLOT=4 (INT4).
                    for (int bi = 0; bi < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); bi++) begin
                        for (int li = 0; li < ITEMS_IN_SLOT; li++) begin
                            int physical_lane;
                            int logical_lane;
                            int physical_row;
                            int physical_col;
                            int logical_bank;
                            physical_lane = (bi*ITEMS_IN_SLOT)+li;
                            physical_row = physical_lane / ARRAY_SIZE;
                            physical_col = physical_lane % ARRAY_SIZE;
                            logical_lane = (physical_col * ARRAY_SIZE) + physical_row;
                            logical_bank = (u_unified_buffer.base_bank + (logical_lane / ITEMS_IN_SLOT))
                                % (NUM_COMPUTE_LANES/ITEMS_IN_SLOT);
                            compute_weights_in[physical_lane] <=
                                u_unified_buffer.bank_dout[logical_bank]
                                    [(COMPUTE_DATA_WIDTH*(logical_lane % ITEMS_IN_SLOT)) +: COMPUTE_DATA_WIDTH];
                        end
                    end
                    if (load_is_weights) begin
                        for (int bi = 0; bi < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); bi++) begin
                            int bank_idx;
                            bank_idx = (u_unified_buffer.base_bank + bi) % (NUM_COMPUTE_LANES/ITEMS_IN_SLOT);
                            for (int li = 0; li < ITEMS_IN_SLOT; li++) begin
                                int lane;
                                lane = (bi*ITEMS_IN_SLOT)+li;
                                sim_block_weights[lane/ARRAY_SIZE][lane%ARRAY_SIZE] <=
                                    u_unified_buffer.bank_dout[bank_idx][(COMPUTE_DATA_WIDTH*li) +: COMPUTE_DATA_WIDTH];
                            end
                        end
                    end else begin
                        if (load_chunk_index == '0) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                sim_block_inputs[row] <= '0;
                                for (int col = 0; col < MAX_BATCH_COUNT; col++) begin
                                    loaded_input_matrix[row][col] <= '0;
                                    sim_block_inputs_matrix[row][col] <= '0;
                                end
                            end
                        end
                        for (int bi = 0; bi < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); bi++) begin
                            int bank_idx;
                            bank_idx = (u_unified_buffer.base_bank + bi) % (NUM_COMPUTE_LANES/ITEMS_IN_SLOT);
                            for (int li = 0; li < ITEMS_IN_SLOT; li++) begin
                                int lane;
                                int global_col;
                                logic signed [COMPUTE_DATA_WIDTH-1:0] lane_val;
                                lane = (bi*ITEMS_IN_SLOT)+li;
                                global_col = load_chunk_index + (lane / ARRAY_SIZE);
                                lane_val = u_unified_buffer.bank_dout[bank_idx][(COMPUTE_DATA_WIDTH*li) +: COMPUTE_DATA_WIDTH];
                                if (lane < ARRAY_SIZE && load_chunk_index == '0)
                                    sim_block_inputs[lane] <= lane_val;
                                if (global_col < input_batch_count) begin
                                    loaded_input_matrix[lane % ARRAY_SIZE][global_col] <= lane_val;
                                    sim_block_inputs_matrix[lane % ARRAY_SIZE][global_col] <= lane_val;
                                end
                            end
                        end
                    end
`else
                    if (load_is_weights) begin
                        for (int lane = 0; lane < NUM_COMPUTE_LANES; lane++) begin
                            int physical_row;
                            int physical_col;
                            int logical_lane;
                            physical_row = lane / ARRAY_SIZE;
                            physical_col = lane % ARRAY_SIZE;
                            logical_lane = (physical_col * ARRAY_SIZE) + physical_row;
                            compute_weights_in[lane] <= mem_to_compute[logical_lane];
                        end
                    end else begin
                        if (load_chunk_index == '0) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                for (int col = 0; col < MAX_BATCH_COUNT; col++)
                                    loaded_input_matrix[row][col] <= '0;
                            end
                        end
                        for (int lane = 0; lane < NUM_COMPUTE_LANES; lane++) begin
                            int global_col;
                            global_col = load_chunk_index + (lane / ARRAY_SIZE);
                            if (global_col < input_batch_count)
                                loaded_input_matrix[lane % ARRAY_SIZE][global_col] <= mem_to_compute[lane];
                        end
                    end
`endif
                    // Pulse only on valid-load cycle so PE weights don't capture invalid/X data.
                    compute_load_en <= load_is_weights;
                    if ((load_chunk_index + ARRAY_SIZE) < input_batch_count && ~load_is_weights) begin
                        buffer_re         <= 1'b0;
                        buffer_compute_en <= 1'b0;
                        load_chunk_index <= load_chunk_index + ARRAY_SIZE;
                        address <= address + ADDRESS_SIZE'(STREAM_CHUNK_WORDS);
                        load_wait_clear <= 1'b1;
                    end else begin
                        buffer_re         <= 1'b0;
                        buffer_compute_en <= 1'b0;
                        load_chunk_index  <= '0;
                        load_wait_clear   <= 1'b0;
                    end
                end
                end
            end

            RESIDUAL_FETCH_STATE: begin
                compute_en        <= 1'b0;
                buffer_re         <= 1'b1;
                buffer_compute_en <= 1'b1;
                compute_load_en   <= 1'b0;
                address           <= residual_source_addr;
                if (buffer_done) begin
`ifdef ICARUS
                    for (int bi = 0; bi < (NUM_COMPUTE_LANES/ITEMS_IN_SLOT); bi++) begin
                        for (int li = 0; li < ITEMS_IN_SLOT; li++) begin
                            residual_in[(bi*ITEMS_IN_SLOT)+li] <=
                                u_unified_buffer.bank_dout[bi][(COMPUTE_DATA_WIDTH*li) +: COMPUTE_DATA_WIDTH];
                        end
                    end
`else
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++) begin
                        residual_in[ci] <= mem_to_compute[ci];
                    end
`endif
                    buffer_re         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                end
            end

            // -----------------------------------------------------------------
            COMPUTE_STATE: begin
                // Legacy direct ReLU path (sized to RELU_SIZE; override
                // RELU_LANES=N^2 to restore full-tile one-shot behavior).
                if (~compute_en && ~quantizer_en && relu_en) begin
                    compute_start <= 1'b0;
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= '0;
                    for (int ci = 0; ci < RELU_SIZE; ci++) begin
                        relu_in[ci] <= mem_to_compute[ci];
                        compute_to_buffer[ci] <= relu_out[ci];
                    end

                // Legacy compute->quantize path
                end else if (compute_en && quantizer_en && ~relu_en) begin
                    compute_start <= compute_en;
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= '0;
                    for (int ci = 0; ci < QUANTIZER_SIZE; ci++) begin
                        quantizer_in[ci] <= compute_stream_out[ci];
                        compute_to_buffer[ci] <= quantizer_out[ci];
                    end

                // Legacy compute->quantize->relu path
                end else if (compute_en && quantizer_en && relu_en) begin
                    compute_start <= compute_en;
                    for (int ci = 0; ci < NUM_COMPUTE_LANES; ci++)
                        compute_to_buffer[ci] <= '0;
                    for (int ci = 0; ci < QUANTIZER_SIZE; ci++)
                        quantizer_in[ci] <= compute_stream_out[ci];
                    for (int ci = 0; ci < RELU_SIZE && ci < QUANTIZER_SIZE; ci++) begin
                        relu_in[ci] <= quantizer_out[ci];
                        compute_to_buffer[ci] <= relu_out[ci];
                    end

                // New blocked-FC accumulate mode:
                // RUN with compute=1, quantize=0, relu=0 accumulates raw PE outputs.
                end else if (compute_en && ~quantizer_en && ~relu_en) begin
`ifdef ICARUS
                    if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                        if (u_pe_array.cycle_count == '0) begin
                            for (int ai = 0; ai < ARRAY_SIZE; ai++)
                                run_capture_sums[ai] <= '0;
                        end
                        if (u_pe_array.cycle_count >= ARRAY_SIZE + 1 &&
                            u_pe_array.cycle_count <  (ARRAY_SIZE + ARRAY_SIZE + 1)) begin
                            int lane_idx;
                            logic signed [31:0] lane_val0;
                            lane_idx = u_pe_array.cycle_count - ARRAY_SIZE - 1;
                            lane_val0 = $signed(u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][lane_idx]);
                            if (^u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][lane_idx] === 1'bx)
                                lane_val0 = 32'sd0;
                            run_capture_sums[lane_idx] <= lane_val0;
                        end
                    end else begin
                        int capture_cycle;
                        capture_cycle = u_pe_array.cycle_count;
                        if ((capture_cycle >= (ARRAY_SIZE + 1)) &&
                            (capture_cycle < ((ARRAY_SIZE * 2) + run_batch_count))) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                int col_idx;
                                logic signed [31:0] lane_val0;
                                col_idx = capture_cycle - ARRAY_SIZE - 1 - row;
                                if ((col_idx >= 0) && (col_idx < run_batch_count)) begin
                                    lane_val0 = $signed(u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][row]);
                                    if (^u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][row] === 1'bx)
                                        lane_val0 = 32'sd0;
                                    if (acc_clear_en)
                                        acc_partial_matrix[row][col_idx] <= lane_val0;
                                    else
                                        acc_partial_matrix[row][col_idx] <= acc_partial_matrix[row][col_idx] + lane_val0;
                                    if (col_idx == 0) begin
                                        if (acc_clear_en)
                                            acc_partial_sums[row] <= lane_val0;
                                        else
                                            acc_partial_sums[row] <= acc_partial_sums[row] + lane_val0;
                                    end
                                end
                            end
                        end
                        if (compute_done) begin
                            logic signed [31:0] lane_val0;
                            lane_val0 = $signed(u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][ARRAY_SIZE-1]);
                            if (^u_pe_array.u_pe_array.accumulators[ARRAY_SIZE-1][ARRAY_SIZE-1] === 1'bx)
                                lane_val0 = 32'sd0;
                            if (acc_clear_en)
                                acc_partial_matrix[ARRAY_SIZE-1][run_batch_count-1] <= lane_val0;
                            else
                                acc_partial_matrix[ARRAY_SIZE-1][run_batch_count-1]
                                    <= acc_partial_matrix[ARRAY_SIZE-1][run_batch_count-1] + lane_val0;
                        end
                    end
`endif
                    if (compute_done_d) begin
`ifndef ICARUS
                        for (int row = 0; row < ARRAY_SIZE; row++) begin
                            logic signed [31:0] lane_val0;
                            for (int col = 0; col < MAX_BATCH_COUNT; col++) begin
                                if (col < run_batch_count) begin
                                    lane_val0 = $signed(compute_stream_out[(col * ARRAY_SIZE) + row]);
                                    if (acc_clear_en)
                                        acc_partial_matrix[row][col] <= lane_val0;
                                    else
                                        acc_partial_matrix[row][col] <= acc_partial_matrix[row][col] + lane_val0;
                                end
                            end
                            if (acc_clear_en)
                                acc_partial_sums[row] <= $signed(compute_stream_out[row]);
                            else
                                acc_partial_sums[row] <= acc_partial_sums[row] + $signed(compute_stream_out[row]);
                            if (acc_clear_en) begin
                                for (int col = 0; col < MAX_BATCH_COUNT; col++)
                                    if (col >= run_batch_count)
                                        acc_partial_matrix[row][col] <= '0;
                            end
                        end
`else
                        if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                logic signed [31:0] lane_val0;
                                lane_val0 = 32'sd0;
                                for (int k = 0; k < ARRAY_SIZE; k++) begin
                                    lane_val0 = lane_val0
                                        + ($signed(sim_block_weights[row][k]) * $signed(loaded_input_matrix[k][0]));
                                end
                                if (acc_clear_en) begin
                                    acc_partial_sums[row] <= lane_val0;
                                    acc_partial_matrix[row][0] <= lane_val0;
                                end else begin
                                    acc_partial_sums[row] <= acc_partial_sums[row] + lane_val0;
                                    acc_partial_matrix[row][0] <= acc_partial_matrix[row][0] + lane_val0;
                                end
                                for (int col = 1; col < MAX_BATCH_COUNT; col++)
                                    acc_partial_matrix[row][col] <= '0;
                            end
                        end else if (acc_clear_en) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                for (int col = 0; col < MAX_BATCH_COUNT; col++)
                                    if (col >= run_batch_count)
                                        acc_partial_matrix[row][col] <= '0;
                            end
                        end
`endif
                        compute_start <= 1'b0;
                        perf_waiting_for_immediate_finalize <= 1'b1;
                    end else if (~compute_start) begin
                        for (int ci = 0; ci < MAX_STREAM_LANES; ci++)
                            compute_stream_in[ci] <= '0;
                        for (int col = 0; col < MAX_BATCH_COUNT; col++) begin
                            if (col < run_batch_count) begin
                                for (int row = 0; row < ARRAY_SIZE; row++)
                                    compute_stream_in[(col * ARRAY_SIZE) + row] <= loaded_input_matrix[row][col];
                            end
                        end
                        compute_start <= 1'b1;
                    end

                // New finalize mode:
                // RUN with compute=0, quantize=1, relu={0|1} writes quantized (and optionally ReLU'd)
                // accumulator values into compute_to_buffer.
                // Narrow (default): stream one output column per cycle through
                // QUANTIZER_SIZE==ARRAY_SIZE. Wide (QUANTIZER_LANES=N^2): one-shot tile.
                end else if (~compute_en && quantizer_en) begin
                    compute_start <= 1'b0;
                    writeback_chunk_index <= '0;
                    writeback_col_index   <= '0;
                    if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                        writeback_chunk_count    <= MAX_BATCH_COUNT_WIDTH'(1);
                        writeback_cols_in_chunk  <= ($clog2(ARRAY_SIZE+1))'(1);
                    end else begin
                        writeback_chunk_count <= (run_batch_count + ARRAY_SIZE - 1) / ARRAY_SIZE;
                        if (run_batch_count < ARRAY_SIZE)
                            writeback_cols_in_chunk <= ($clog2(ARRAY_SIZE+1))'(run_batch_count);
                        else
                            writeback_cols_in_chunk <= ($clog2(ARRAY_SIZE+1))'(ARRAY_SIZE);
                    end
                    if (requant_enable_latched) begin
                        writeback_wait_clear <= 1'b1;
                        writeback_pipe_fill_cnt <= QUANTIZER_PIPE_DEPTH[3:0];
                        if (REQUANT_NARROW) begin
                            clear_compute_to_buffer();
                            prepare_writeback_column(0, 0);
                        end else if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                            for (int ai = 0; ai < QUANTIZER_SIZE; ai++)
                                quantizer_in[ai] <= '0;
                            for (int ai = 0; ai < ARRAY_SIZE && ai < QUANTIZER_SIZE; ai++) begin
`ifdef ICARUS
                                logic signed [31:0] acc_val;
                                logic signed [31:0] residual_val;
                                acc_val = acc_partial_sums[ai];
                                if (^acc_partial_sums[ai] === 1'bx)
                                    acc_val = 32'sd0;
                                residual_val = residual_en ? $signed(residual_in[ai]) : 32'sd0;
                                if (residual_en && (^residual_in[ai] === 1'bx))
                                    residual_val = 32'sd0;
                                acc_val = acc_val + residual_val;
                                quantizer_in[ai] <= acc_val[ACCUMULATOR_DATA_WIDTH-1:0];
`else
                                logic signed [31:0] acc_val;
                                logic signed [31:0] residual_val;
                                acc_val = acc_partial_sums[ai];
                                residual_val = residual_en ? $signed(residual_in[ai]) : 32'sd0;
                                acc_val = acc_val + residual_val;
                                quantizer_in[ai] <= acc_val[ACCUMULATOR_DATA_WIDTH-1:0];
`endif
                            end
                        end else begin
                            prepare_writeback_chunk(0);
                        end
                    end else begin
                        writeback_wait_clear <= 1'b0;
                        writeback_pipe_fill_cnt <= 4'd0;
                        if (run_batch_count == MAX_BATCH_COUNT_WIDTH'(1)) begin
                            for (int ai = 0; ai < NUM_COMPUTE_LANES; ai++)
                                compute_to_buffer[ai] <= '0;
                            for (int ai = 0; ai < ARRAY_SIZE; ai++) begin
`ifdef ICARUS
                                logic signed [31:0] acc_val;
                                logic signed [31:0] residual_val;
                                acc_val = acc_partial_sums[ai];
                                if (^acc_partial_sums[ai] === 1'bx)
                                    acc_val = 32'sd0;
                                residual_val = residual_en ? $signed(residual_in[ai]) : 32'sd0;
                                if (residual_en && (^residual_in[ai] === 1'bx))
                                    residual_val = 32'sd0;
                                acc_val = acc_val + residual_val;
                                if (relu_en && (quantize_clip_int4(acc_val) < 0))
                                    compute_to_buffer[ai] <= quantize_clip_int4(acc_val) >>> ALPHA;
                                else
                                    compute_to_buffer[ai] <= quantize_clip_int4(acc_val);
`else
                                logic signed [31:0] acc_val;
                                logic signed [31:0] residual_val;
                                acc_val = acc_partial_sums[ai];
                                residual_val = residual_en ? $signed(residual_in[ai]) : 32'sd0;
                                acc_val = acc_val + residual_val;
                                if (relu_en && (quantize_clip_int4(acc_val) < 0))
                                    compute_to_buffer[ai] <= quantize_clip_int4(acc_val) >>> ALPHA;
                                else
                                    compute_to_buffer[ai] <= quantize_clip_int4(acc_val);
`endif
                            end
                        end else begin
                            prepare_writeback_chunk(0);
                        end
                    end
                end else begin
                    compute_start <= 1'b0;
                end
            end

            // -----------------------------------------------------------------
            COMPUTE_WRITEBACK_STATE: begin
                buffer_re         <= 1'b0;
                buffer_fifo_en    <= 1'b0;
                buffer_store_en   <= 1'b0;
                address           <= compute_result_addr + (writeback_chunk_index * STREAM_CHUNK_WORDS);
                if (requant_finalize_enable && writeback_wait_clear) begin
                    if (writeback_pipe_fill_cnt != 4'd0) begin
                        // Registered quantizer: countdown after presenting
                        // quantizer_in before capture (DEPTH cycles of wait).
                        writeback_pipe_fill_cnt <= writeback_pipe_fill_cnt - 4'd1;
                        buffer_we         <= 1'b0;
                        buffer_compute_en <= 1'b0;
                    end else if (REQUANT_NARROW) begin
                        // Capture current column only. Presenting the next
                        // column's inputs in this same cycle would keep a
                        // registered quantizer on the old operand, so arm
                        // another fill countdown before the next capture.
                        capture_writeback_column(writeback_col_index);
                        if ((writeback_col_index + 1'b1) < writeback_cols_in_chunk) begin
                            writeback_col_index <= writeback_col_index + 1'b1;
                            prepare_writeback_column(
                                writeback_chunk_index,
                                writeback_col_index + 1
                            );
                            writeback_pipe_fill_cnt <= QUANTIZER_PIPE_DEPTH[3:0];
                        end else begin
                            writeback_wait_clear <= 1'b0;
                            writeback_col_index  <= '0;
                        end
                        buffer_we         <= 1'b0;
                        buffer_compute_en <= 1'b0;
                    end else begin
                        for (int ai = 0; ai < QUANTIZER_SIZE; ai++) begin
                            if (relu_en)
                                compute_to_buffer[ai] <= apply_leaky_relu(quantizer_out[ai]);
                            else
                                compute_to_buffer[ai] <= quantizer_out[ai];
                        end
                        buffer_we         <= 1'b0;
                        buffer_compute_en <= 1'b0;
                        if (!buffer_done)
                            writeback_wait_clear <= 1'b0;
                    end
                end else if (writeback_wait_clear) begin
                    buffer_we         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                    if (!buffer_done)
                        writeback_wait_clear <= 1'b0;
                end else if (buffer_done) begin
                    buffer_we         <= 1'b0;
                    buffer_compute_en <= 1'b0;
                    if ((writeback_chunk_index + 1'b1) < writeback_chunk_count) begin
                        int next_chunk;
                        int remaining;
                        next_chunk = writeback_chunk_index + 1;
                        writeback_chunk_index <= writeback_chunk_index + 1'b1;
                        writeback_wait_clear  <= 1'b1;
                        writeback_col_index   <= '0;
                        writeback_pipe_fill_cnt <= requant_enable_latched ? QUANTIZER_PIPE_DEPTH[3:0] : 4'd0;
                        remaining = run_batch_count - (next_chunk * ARRAY_SIZE);
                        if (remaining < ARRAY_SIZE)
                            writeback_cols_in_chunk <= ($clog2(ARRAY_SIZE+1))'(remaining);
                        else
                            writeback_cols_in_chunk <= ($clog2(ARRAY_SIZE+1))'(ARRAY_SIZE);
                        if (REQUANT_NARROW && requant_enable_latched) begin
                            clear_compute_to_buffer();
                            prepare_writeback_column(next_chunk, 0);
                        end else begin
                            prepare_writeback_chunk(next_chunk);
                        end
                    end else begin
                        writeback_chunk_index <= '0;
                        writeback_chunk_count <= MAX_BATCH_COUNT_WIDTH'(1);
                        writeback_wait_clear  <= 1'b0;
                        writeback_col_index   <= '0;
                        writeback_pipe_fill_cnt <= 4'd0;
                    end
                end else begin
                    buffer_we         <= 1'b1;
                    buffer_compute_en <= 1'b1;
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
                // Legacy: base address lives in the opcode word's high bits.
                // Ext-addr: it was already latched in EXT_ADDR_LATCH_STATE
                // (do not overwrite here).
                if (!EXT_ADDR_EN)
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
        end else if (perf_stream_active && ~tx_full) begin
            tx_we    <= 1'b1;
            tx_wdata <= perf_snapshot[191 - (perf_stream_idx * 8) -: 8];
            if (perf_stream_idx == 5'd23) begin
                perf_stream_active <= 1'b0;
                perf_stream_idx    <= '0;
            end else begin
                perf_stream_idx <= perf_stream_idx + 1'b1;
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
