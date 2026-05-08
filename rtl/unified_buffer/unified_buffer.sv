
`timescale 1ns/1ps

module unified_buffer #(
	parameter BUFFER_SIZE 	     = 1024, // The amount of words in the buffer
	parameter BUFFER_WORD_SIZE   = 16,   // Number of bits stored in each cell
	parameter FIFO_DATA_WIDTH    = 8,    // Number of bits recieved/sent from/to fifos
	parameter COMPUTE_DATA_WIDTH = 4,  // Number of bits recieved/sent from/to compute unit
	parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE),
	parameter ARRAY_SIZE         = 8,
	parameter NUM_COMPUTE_LANES  = ARRAY_SIZE*ARRAY_SIZE,
	parameter STORE_DATA_WIDTH   = 16
    ) (
	input  logic clk, we, re, compute_en, fifo_en, store_en,
	output logic 			      done,
	input  logic 			      section,  // Used for fifo where 0 top/1 bot
	input  logic [ADDRESS_SIZE-1:0]	      address,
	input  logic [FIFO_DATA_WIDTH-1:0]    fifo_in,
	output logic [FIFO_DATA_WIDTH-1:0]    fifo_out,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0] compute_in [NUM_COMPUTE_LANES-1:0], 
	output logic signed [COMPUTE_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0],
	input  logic [STORE_DATA_WIDTH-1:0]   store_in,
	output logic [STORE_DATA_WIDTH-1:0]   store_out
    ); 
    
    localparam int ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;
    localparam int BANKS         = NUM_COMPUTE_LANES/ITEMS_IN_SLOT;
    localparam int BANK_BITS     = $clog2(BANKS);
    localparam int BANK_DEPTH    = BUFFER_SIZE / BANKS;
    localparam int BANK_ADDR_W   = $clog2(BANK_DEPTH);

    // Banked BRAM to provide enough read/write bandwidth for compute lanes.
    // Each bank is an explicit Xilinx block RAM primitive so implementation does
    // not fall back to LUT/FF memory heuristics.
    logic [BUFFER_WORD_SIZE-1:0] compute_word [BANKS-1:0];
    logic [BUFFER_WORD_SIZE-1:0] bank_dout    [BANKS-1:0];
    logic [BUFFER_WORD_SIZE-1:0] bank_din     [BANKS-1:0];
    logic [BANK_ADDR_W-1:0]      bank_waddr   [BANKS-1:0];
    logic [BANK_ADDR_W-1:0]      bank_raddr   [BANKS-1:0];
    logic [1:0]                  bank_we      [BANKS-1:0];
    logic                        bank_re      [BANKS-1:0];
    logic [BANK_BITS-1:0]        read_bank_sel_d;
    logic                        read_section_d;
    logic                        read_fifo_d;
    logic                        read_store_d;
    logic                        read_compute_d;
    logic [BANK_BITS-1:0]        base_bank;
    logic [BANK_ADDR_W-1:0]      base_row;
    logic [NUM_COMPUTE_LANES*COMPUTE_DATA_WIDTH-1:0] compute_out_flat;

    function automatic [BANK_ADDR_W-1:0] row_for_bank(
        input int bank_idx,
        input [BANK_ADDR_W-1:0] row,
        input [BANK_BITS-1:0] bank_sel
    );
        if (bank_idx < bank_sel)
            row_for_bank = row + 1'b1;
        else
            row_for_bank = row;
    endfunction

    always_comb begin
`ifdef ICARUS
        base_bank = address % BANKS;
        base_row  = address / BANKS;
`else
        base_bank = address[BANK_BITS-1:0];
        base_row  = address[ADDRESS_SIZE-1:BANK_BITS];
`endif
    end

    integer init_i;
    initial begin
        done = 1'b0;
        fifo_out = '0;
        store_out = '0;
        read_bank_sel_d = '0;
        read_section_d = 1'b0;
        read_fifo_d = 1'b0;
        read_store_d = 1'b0;
        read_compute_d = 1'b0;
        compute_out_flat = '0;
    end

    genvar gi, gj;
    generate
        for (gi = 0; gi < NUM_COMPUTE_LANES; gi++) begin: gen_unpack_compute_out
            assign compute_out[gi] = compute_out_flat[(COMPUTE_DATA_WIDTH*gi) +: COMPUTE_DATA_WIDTH];
        end
    endgenerate

    generate
        for (gi = 0; gi < BANKS; gi++) begin: gen_pack
            for (gj = 0; gj < ITEMS_IN_SLOT; gj++) begin: gen_pack_lanes
                assign compute_word[gi][(COMPUTE_DATA_WIDTH*gj) +: COMPUTE_DATA_WIDTH] =
                    compute_in[gj + gi*ITEMS_IN_SLOT];
            end
        end
    endgenerate

    always_comb begin
        for (int i = 0; i < BANKS; i++) begin
            bank_waddr[i] = row_for_bank(i, base_row, base_bank);
            bank_raddr[i] = row_for_bank(i, base_row, base_bank);
            bank_din[i]   = compute_word[i];
            bank_we[i]    = 2'b00;
            bank_re[i]    = 1'b0;
        end

        if (we) begin
            if (compute_en) begin
                for (int i = 0; i < BANKS; i++) begin
                    bank_we[i] = 2'b11;
                    bank_din[i] = compute_word[i];
                end
            end else if (fifo_en) begin
                bank_we[base_bank] = section ? 2'b10 : 2'b01;
                bank_din[base_bank] = section
                    ? {fifo_in, 8'h00}
                    : {8'h00, fifo_in};
            end else if (store_en) begin
                bank_we[base_bank] = 2'b11;
                bank_din[base_bank] = store_in;
            end
        end

        if (re) begin
            if (compute_en) begin
                for (int i = 0; i < BANKS; i++) begin
                    bank_re[i] = 1'b1;
                end
            end else begin
                bank_re[base_bank] = fifo_en | store_en;
            end
        end
    end

`ifdef ICARUS
    // Icarus behavioral replacement for XPM SDP RAM.
    logic [BUFFER_WORD_SIZE-1:0] bank_mem [BANKS-1:0][BANK_DEPTH-1:0];
    integer bmi, bmr;
    initial begin
        for (bmi = 0; bmi < BANKS; bmi = bmi + 1) begin
            for (bmr = 0; bmr < BANK_DEPTH; bmr = bmr + 1)
                bank_mem[bmi][bmr] = '0;
            bank_dout[bmi] = '0;
        end
    end

    always_ff @(posedge clk) begin
        for (int bi = 0; bi < BANKS; bi++) begin
            // Byte write semantics compatible with XPM BYTE_WRITE_WIDTH_A=8.
            if (bank_we[bi][0])
                bank_mem[bi][bank_waddr[bi]][7:0] <= bank_din[bi][7:0];
            if (bank_we[bi][1])
                bank_mem[bi][bank_waddr[bi]][15:8] <= bank_din[bi][15:8];
            // 1-cycle read latency
            if (bank_re[bi])
                bank_dout[bi] <= bank_mem[bi][bank_raddr[bi]];
        end
    end
`else
    genvar bi;
    generate
        for (bi = 0; bi < BANKS; bi++) begin: gen_bram
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(BANK_ADDR_W),
                .ADDR_WIDTH_B(BANK_ADDR_W),
                .AUTO_SLEEP_TIME(0),
                .BYTE_WRITE_WIDTH_A(8),
                .CASCADE_HEIGHT(0),
                .CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE("none"),
                .MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"),
                .MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(BANK_DEPTH*BUFFER_WORD_SIZE),
                .MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(BUFFER_WORD_SIZE),
                .READ_LATENCY_B(1),
                .READ_RESET_VALUE_B("0"),
                .RST_MODE_A("SYNC"),
                .RST_MODE_B("SYNC"),
                .SIM_ASSERT_CHK(0),
                .USE_EMBEDDED_CONSTRAINT(0),
                .USE_MEM_INIT(1),
                .WAKEUP_TIME("disable_sleep"),
                .WRITE_DATA_WIDTH_A(BUFFER_WORD_SIZE),
                .WRITE_MODE_B("no_change")
            ) u_bank_bram (
                .clka(clk),
                .clkb(clk),
                .ena(1'b1),
                .enb(bank_re[bi]),
                .addra(bank_waddr[bi]),
                .addrb(bank_raddr[bi]),
                .dina(bank_din[bi]),
                .wea(bank_we[bi]),
                .doutb(bank_dout[bi]),
                .injectdbiterra(1'b0),
                .injectsbiterra(1'b0),
                .regceb(1'b1),
                .rstb(1'b0),
                .sleep(1'b0)
            );
        end
    endgenerate
`endif

    always_ff @(posedge clk) begin
        done <= we | re;

        read_bank_sel_d <= base_bank;
        read_section_d <= section;
        read_fifo_d <= re && fifo_en;
        read_store_d <= re && store_en;
        read_compute_d <= re && compute_en;

        if (read_compute_d) begin
`ifdef ICARUS
            if (ITEMS_IN_SLOT == 4 && COMPUTE_DATA_WIDTH == 4) begin
                for (int i = 0; i < BANKS; i++) begin
                    compute_out_flat[(COMPUTE_DATA_WIDTH*((i*4)+0)) +: COMPUTE_DATA_WIDTH] <= bank_dout[i][3:0];
                    compute_out_flat[(COMPUTE_DATA_WIDTH*((i*4)+1)) +: COMPUTE_DATA_WIDTH] <= bank_dout[i][7:4];
                    compute_out_flat[(COMPUTE_DATA_WIDTH*((i*4)+2)) +: COMPUTE_DATA_WIDTH] <= bank_dout[i][11:8];
                    compute_out_flat[(COMPUTE_DATA_WIDTH*((i*4)+3)) +: COMPUTE_DATA_WIDTH] <= bank_dout[i][15:12];
                end
            end else begin
                for (int i = 0; i < BANKS; i++) begin
                    for (int j = 0; j < ITEMS_IN_SLOT; j++) begin
                        compute_out_flat[(COMPUTE_DATA_WIDTH*(j + i*ITEMS_IN_SLOT)) +: COMPUTE_DATA_WIDTH]
                            <= bank_dout[i][(COMPUTE_DATA_WIDTH*j) +: COMPUTE_DATA_WIDTH];
                    end
                end
            end
`else
            for (int i = 0; i < BANKS; i++) begin
                for (int j = 0; j < ITEMS_IN_SLOT; j++) begin
                    compute_out_flat[(COMPUTE_DATA_WIDTH*(j + i*ITEMS_IN_SLOT)) +: COMPUTE_DATA_WIDTH]
                        <= bank_dout[i][(COMPUTE_DATA_WIDTH*j) +: COMPUTE_DATA_WIDTH];
                end
            end
`endif
        end

        if (read_fifo_d) begin
            fifo_out <= read_section_d
                ? bank_dout[read_bank_sel_d][FIFO_DATA_WIDTH +: FIFO_DATA_WIDTH]
                : bank_dout[read_bank_sel_d][0 +: FIFO_DATA_WIDTH];
        end

        if (read_store_d) begin
            store_out <= bank_dout[read_bank_sel_d][STORE_DATA_WIDTH-1:0];
        end

    end

endmodule: unified_buffer
