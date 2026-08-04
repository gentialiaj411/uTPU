`timescale 1ns/1ps

// Out-of-context BSTORE write-arm widen estimate.
// Models a WIDTH-word skid filled from a 1-word/cycle instr source, then a
// WIDTH-way parallel store write (sequential addresses). Measurement-only —
// not wired into top.sv.
module bstore_wide_arm_ooc #(
    parameter int WIDTH = 1,
    parameter int ADDR_W = 12,
    parameter int DATA_W = 16
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 src_valid,
    input  logic [DATA_W-1:0]    src_data,
    input  logic                 fire_write,
    input  logic [ADDR_W-1:0]    base_addr,
    output logic                 skid_full,
    output logic                 busy,
    output logic [WIDTH-1:0]     bank_we,
    output logic [ADDR_W-1:0]    bank_waddr [WIDTH-1:0],
    output logic [DATA_W-1:0]    bank_wdata [WIDTH-1:0]
);
    logic [DATA_W-1:0] skid [WIDTH-1:0];
    logic [$clog2(WIDTH+1)-1:0] fill;
    logic writing;

    assign skid_full = (fill == WIDTH[$clog2(WIDTH+1)-1:0]);
    assign busy = writing;

    integer i;
    always_ff @(posedge clk) begin
        if (rst) begin
            fill <= '0;
            writing <= 1'b0;
            bank_we <= '0;
            for (i = 0; i < WIDTH; i++) begin
                skid[i] <= '0;
                bank_waddr[i] <= '0;
                bank_wdata[i] <= '0;
            end
        end else begin
            bank_we <= '0;
            if (writing) begin
                writing <= 1'b0;
                fill <= '0;
            end else if (fire_write && skid_full) begin
                writing <= 1'b1;
                for (i = 0; i < WIDTH; i++) begin
                    bank_we[i] <= 1'b1;
                    bank_waddr[i] <= base_addr + i[ADDR_W-1:0];
                    bank_wdata[i] <= skid[i];
                end
            end else if (src_valid && !skid_full) begin
                skid[fill] <= src_data;
                fill <= fill + 1'b1;
            end
        end
    end
endmodule
