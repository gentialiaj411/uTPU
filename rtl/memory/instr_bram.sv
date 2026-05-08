`timescale 1ns/1ps

// Synchronous dual-port instruction BRAM.
// Write port: used during upload phase to store instructions.
// Read port: 1-cycle latency — data appears on rd_data the cycle after rd_en is asserted.
module instr_bram #(
    parameter DEPTH  = 1024,
    parameter WIDTH  = 16,
    parameter ADDR_W = $clog2(DEPTH)
) (
    input  logic              clk,
    // Write port
    input  logic              wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic [WIDTH-1:0]  wr_data,
    // Read port
    input  logic              rd_en,
    input  logic [ADDR_W-1:0] rd_addr,
    output logic [WIDTH-1:0]  rd_data
);

    (* ram_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    always_ff @(posedge clk) begin
        if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule: instr_bram
