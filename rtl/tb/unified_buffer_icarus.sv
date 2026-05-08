`timescale 1ns/1ps

// Icarus-only behavioral drop-in for unified_buffer.
// Keeps the exact module name/interface used by top.sv.
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
    input  logic signed [COMPUTE_DATA_WIDTH-1:0] compute_in [NUM_COMPUTE_LANES-1:0],
    output logic signed [COMPUTE_DATA_WIDTH-1:0] compute_out [NUM_COMPUTE_LANES-1:0],
    input  logic [STORE_DATA_WIDTH-1:0]       store_in,
    output logic [STORE_DATA_WIDTH-1:0]       store_out
);
    localparam int ITEMS_IN_SLOT = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH;
    localparam int BANKS         = NUM_COMPUTE_LANES/ITEMS_IN_SLOT;
    localparam int BANK_BITS     = $clog2(BANKS);
    localparam int BANK_DEPTH    = BUFFER_SIZE / BANKS;
    localparam int BANK_ADDR_W   = $clog2(BANK_DEPTH);

    logic [BUFFER_WORD_SIZE-1:0] mem [0:BUFFER_SIZE-1];
    logic [BUFFER_WORD_SIZE-1:0] compute_word [0:BANKS-1];
    logic [BUFFER_WORD_SIZE-1:0] read_bank_word [0:BANKS-1];
    logic [ADDRESS_SIZE-1:0] direct_addr;
    logic [ADDRESS_SIZE-1:0] bank_addr [0:BANKS-1];

    integer mi, ci, bi, lj;
    initial begin
        done = 1'b0;
        fifo_out = '0;
        store_out = '0;
        for (mi = 0; mi < BUFFER_SIZE; mi = mi + 1) mem[mi] = '0;
        for (ci = 0; ci < NUM_COMPUTE_LANES; ci = ci + 1) compute_out[ci] = '0;
        for (bi = 0; bi < BANKS; bi = bi + 1) read_bank_word[bi] = '0;
    end

    always_comb begin
        direct_addr = address;
        for (bi = 0; bi < BANKS; bi = bi + 1) begin
            bank_addr[bi] = direct_addr + bi[ADDRESS_SIZE-1:0];
            if (ITEMS_IN_SLOT == 4 && COMPUTE_DATA_WIDTH == 4) begin
                compute_word[bi][3:0]    = compute_in[(bi*4)+0];
                compute_word[bi][7:4]    = compute_in[(bi*4)+1];
                compute_word[bi][11:8]   = compute_in[(bi*4)+2];
                compute_word[bi][15:12]  = compute_in[(bi*4)+3];
            end else begin
                compute_word[bi] = '0;
                for (lj = 0; lj < ITEMS_IN_SLOT; lj = lj + 1)
                    compute_word[bi][(COMPUTE_DATA_WIDTH*lj)+:COMPUTE_DATA_WIDTH] =
                        compute_in[lj + bi*ITEMS_IN_SLOT];
            end
        end
    end

    always_ff @(posedge clk) begin
        done <= we | re;

        if (we) begin
            if (compute_en) begin
                for (bi = 0; bi < BANKS; bi = bi + 1) begin
                    mem[bank_addr[bi]] <= compute_word[bi];
                end
            end else if (fifo_en) begin
                if (!section)
                    mem[direct_addr][FIFO_DATA_WIDTH-1:0] <= fifo_in;
                else
                    mem[direct_addr][BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH] <= fifo_in;
            end else if (store_en) begin
                mem[direct_addr][STORE_DATA_WIDTH-1:0] <= store_in;
            end
        end

        if (re) begin
            if (compute_en) begin
                for (bi = 0; bi < BANKS; bi = bi + 1)
                    read_bank_word[bi] <= mem[bank_addr[bi]];
                for (bi = 0; bi < BANKS; bi = bi + 1) begin
                    if (ITEMS_IN_SLOT == 4 && COMPUTE_DATA_WIDTH == 4) begin
                        compute_out[(bi*4)+0] <= read_bank_word[bi][3:0];
                        compute_out[(bi*4)+1] <= read_bank_word[bi][7:4];
                        compute_out[(bi*4)+2] <= read_bank_word[bi][11:8];
                        compute_out[(bi*4)+3] <= read_bank_word[bi][15:12];
                    end else begin
                        for (lj = 0; lj < ITEMS_IN_SLOT; lj = lj + 1) begin
                            compute_out[lj + bi*ITEMS_IN_SLOT] <=
                                read_bank_word[bi][(COMPUTE_DATA_WIDTH*lj)+:COMPUTE_DATA_WIDTH];
                        end
                    end
                end
            end else if (fifo_en) begin
                if (!section)
                    fifo_out <= mem[direct_addr][FIFO_DATA_WIDTH-1:0];
                else
                    fifo_out <= mem[direct_addr][BUFFER_WORD_SIZE-1:FIFO_DATA_WIDTH];
            end else if (store_en) begin
                store_out <= mem[direct_addr][STORE_DATA_WIDTH-1:0];
            end
        end
    end
endmodule
