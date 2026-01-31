/*
*  Unified Buffer - Iverilog Compatible Version
*  Fixed constant selects in always_* processes
*  Fixed unpacked array ports -> packed vectors for iverilog compatibility
*/

`timescale 1ns/1ps

module unified_buffer #(
    parameter BUFFER_SIZE        = 1024,
    parameter BUFFER_WORD_SIZE   = 16,
    parameter FIFO_DATA_WIDTH    = 8,
    parameter COMPUTE_DATA_WIDTH = 4,
    parameter ADDRESS_SIZE       = $clog2(BUFFER_SIZE),
    parameter ARRAY_SIZE         = 8,
    parameter NUM_COMPUTE_LANES  = ARRAY_SIZE*ARRAY_SIZE,
    parameter STORE_DATA_WIDTH   = 16
) (
    input  logic clk, we, re, compute_en, fifo_en, store_en,
    output logic                              done,
    input  logic                              section,
    input  logic [ADDRESS_SIZE-1:0]           address,
    input  logic [FIFO_DATA_WIDTH-1:0]        fifo_in,
    output logic [FIFO_DATA_WIDTH-1:0]        fifo_out,
    // IVERILOG FIX: Changed from unpacked arrays to packed vectors for port compatibility
    input  logic [NUM_COMPUTE_LANES*COMPUTE_DATA_WIDTH-1:0] compute_in_packed, 
    output logic [NUM_COMPUTE_LANES*COMPUTE_DATA_WIDTH-1:0] compute_out_packed,
    input  logic [STORE_DATA_WIDTH-1:0]       store_in,
    output logic [STORE_DATA_WIDTH-1:0]       store_out
);

    // Internal unpacked arrays for convenience
    logic signed [COMPUTE_DATA_WIDTH-1:0] compute_in [NUM_COMPUTE_LANES-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0];

    // Pack/unpack compute arrays
    genvar gi_pack;
    generate
        for (gi_pack = 0; gi_pack < NUM_COMPUTE_LANES; gi_pack++) begin: gen_pack_io
            assign compute_in[gi_pack] = compute_in_packed[gi_pack*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH];
            assign compute_out_packed[gi_pack*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH] = compute_out[gi_pack];
        end
    endgenerate 
    
    localparam int ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;
    localparam int BANKS         = NUM_COMPUTE_LANES/ITEMS_IN_SLOT;
    localparam int BANK_BITS     = $clog2(BANKS);
    localparam int BANK_DEPTH    = BUFFER_SIZE / BANKS;
    localparam int BANK_ADDR_W   = $clog2(BANK_DEPTH);

    // Banked BRAM
    (* ram_style = "block" *) logic [BUFFER_WORD_SIZE-1:0] mem [BANKS-1:0][BANK_DEPTH-1:0];
    logic [BUFFER_WORD_SIZE-1:0] compute_word [BANKS-1:0];
    logic [BANK_BITS-1:0]        base_bank;
    logic [BANK_ADDR_W-1:0]      base_row;

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

    // Combinational address decode
    assign base_bank = address[BANK_BITS-1:0];
    assign base_row  = address[ADDRESS_SIZE-1:BANK_BITS];

    // Pack compute_in into words
    genvar gi, gj;
    generate
        for (gi = 0; gi < BANKS; gi++) begin: gen_pack
            for (gj = 0; gj < ITEMS_IN_SLOT; gj++) begin: gen_pack_lanes
                assign compute_word[gi][(COMPUTE_DATA_WIDTH*gj) +: COMPUTE_DATA_WIDTH] =
                    compute_in[gj + gi*ITEMS_IN_SLOT];
            end
        end
    endgenerate

    // Debug: track last store for verification
    logic [15:0] debug_last_store_addr;
    logic [15:0] debug_last_store_val;
    logic        debug_compute_read_happened;

    always_ff @(posedge clk) begin
        done <= we | re;
        debug_compute_read_happened <= 1'b0;

        if (we) begin
            if (compute_en) begin
                for (int i = 0; i < BANKS; i++) begin
                    mem[i][row_for_bank(i, base_row, base_bank)] <= compute_word[i];
                end
            end else if (fifo_en) begin
                // Iverilog compatible: explicit if/else instead of case with part select
                if (section == 1'b0)
                    mem[base_bank][base_row][FIFO_DATA_WIDTH-1:0] <= fifo_in;
                else
                    mem[base_bank][base_row][BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH] <= fifo_in;
            end else if (store_en) begin 
                mem[base_bank][base_row][STORE_DATA_WIDTH-1:0] <= store_in;
                debug_last_store_addr <= {base_row, base_bank};
                debug_last_store_val <= store_in;
                // Debug: print first few stores
                if (address < 16'h0010 || (address >= 16'h0080 && address < 16'h0090))
                    $display("[BUFFER] STORE: addr=%0d (bank=%0d, row=%0d) val=0x%04X", 
                        address, base_bank, base_row, store_in);
            end

        end else if (re) begin
            if (compute_en) begin
                $display("[BUFFER] COMPUTE_EN READ: addr=%0d (bank=%0d, row=%0d)", address, base_bank, base_row);
                debug_compute_read_happened <= 1'b1;
                // IVERILOG FIX: Use blocking assignments for simulation
                // The original non-blocking version didn't work in iverilog for unpacked arrays
                // Bank 0-3 (explicit unroll for debugging)
                compute_out[0]  = mem[0][row_for_bank(0, base_row, base_bank)][3:0];
                compute_out[1]  = mem[0][row_for_bank(0, base_row, base_bank)][7:4];
                compute_out[2]  = mem[0][row_for_bank(0, base_row, base_bank)][11:8];
                compute_out[3]  = mem[0][row_for_bank(0, base_row, base_bank)][15:12];
                compute_out[4]  = mem[1][row_for_bank(1, base_row, base_bank)][3:0];
                compute_out[5]  = mem[1][row_for_bank(1, base_row, base_bank)][7:4];
                compute_out[6]  = mem[1][row_for_bank(1, base_row, base_bank)][11:8];
                compute_out[7]  = mem[1][row_for_bank(1, base_row, base_bank)][15:12];
                compute_out[8]  = mem[2][row_for_bank(2, base_row, base_bank)][3:0];
                compute_out[9]  = mem[2][row_for_bank(2, base_row, base_bank)][7:4];
                compute_out[10] = mem[2][row_for_bank(2, base_row, base_bank)][11:8];
                compute_out[11] = mem[2][row_for_bank(2, base_row, base_bank)][15:12];
                compute_out[12] = mem[3][row_for_bank(3, base_row, base_bank)][3:0];
                compute_out[13] = mem[3][row_for_bank(3, base_row, base_bank)][7:4];
                compute_out[14] = mem[3][row_for_bank(3, base_row, base_bank)][11:8];
                compute_out[15] = mem[3][row_for_bank(3, base_row, base_bank)][15:12];
                // Bank 4-15 (loop for rest)
                for (int i = 4; i < BANKS; i++) begin
                    compute_out[0 + i*4] = mem[i][row_for_bank(i, base_row, base_bank)][3:0];
                    compute_out[1 + i*4] = mem[i][row_for_bank(i, base_row, base_bank)][7:4];
                    compute_out[2 + i*4] = mem[i][row_for_bank(i, base_row, base_bank)][11:8];
                    compute_out[3 + i*4] = mem[i][row_for_bank(i, base_row, base_bank)][15:12];
                end
                // Debug: print what we're reading from first 4 banks
                $display("[BUFFER]   Bank 0: mem[0][%0d] = 0x%04X -> [%0d,%0d,%0d,%0d]", 
                    row_for_bank(0, base_row, base_bank), mem[0][row_for_bank(0, base_row, base_bank)],
                    $signed(compute_out[0]), $signed(compute_out[1]),
                    $signed(compute_out[2]), $signed(compute_out[3]));
                $display("[BUFFER]   Bank 1: mem[1][%0d] = 0x%04X -> [%0d,%0d,%0d,%0d]", 
                    row_for_bank(1, base_row, base_bank), mem[1][row_for_bank(1, base_row, base_bank)],
                    $signed(compute_out[4]), $signed(compute_out[5]),
                    $signed(compute_out[6]), $signed(compute_out[7]));
            end else if (fifo_en) begin
                if (section == 1'b0)
                    fifo_out <= mem[base_bank][base_row][FIFO_DATA_WIDTH-1:0];
                else
                    fifo_out <= mem[base_bank][base_row][BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH];
            end else if (store_en) begin
                store_out <= mem[base_bank][base_row][STORE_DATA_WIDTH-1:0];
            end
        end
    end

endmodule: unified_buffer
