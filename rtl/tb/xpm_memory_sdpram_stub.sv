`timescale 1ns/1ps

module xpm_memory_sdpram #(
    parameter integer ADDR_WIDTH_A = 9,
    parameter integer ADDR_WIDTH_B = 9,
    parameter integer AUTO_SLEEP_TIME = 0,
    parameter integer BYTE_WRITE_WIDTH_A = 8,
    parameter integer CASCADE_HEIGHT = 0,
    parameter        CLOCKING_MODE = "common_clock",
    parameter        ECC_MODE = "no_ecc",
    parameter        MEMORY_INIT_FILE = "none",
    parameter        MEMORY_INIT_PARAM = "0",
    parameter        MEMORY_OPTIMIZATION = "true",
    parameter        MEMORY_PRIMITIVE = "block",
    parameter integer READ_DATA_WIDTH_B = 16,
    parameter integer READ_LATENCY_B = 1,
    parameter        READ_RESET_VALUE_B = "0",
    parameter        RST_MODE_A = "SYNC",
    parameter        RST_MODE_B = "SYNC",
    parameter integer SIM_ASSERT_CHK = 0,
    parameter integer USE_EMBEDDED_CONSTRAINT = 0,
    parameter integer USE_MEM_INIT = 1,
    parameter        WAKEUP_TIME = "disable_sleep",
    parameter integer WRITE_DATA_WIDTH_A = 16,
    parameter        WRITE_MODE_B = "no_change",
    parameter integer MEMORY_SIZE = 8192,
    parameter integer MESSAGE_CONTROL = 0
) (
    input  wire                       clka,
    input  wire [ADDR_WIDTH_A-1:0]    addra,
    input  wire [WRITE_DATA_WIDTH_A-1:0] dina,
    input  wire                       ena,
    input  wire [(WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A)-1:0] wea,
    input  wire                       clkb,
    input  wire [ADDR_WIDTH_B-1:0]    addrb,
    input  wire                       enb,
    output reg  [READ_DATA_WIDTH_B-1:0] doutb,
    input  wire                       rstb,
    input  wire                       sleep,
    input  wire                       regceb,
    input  wire [0:0]                 injectsbiterra,
    input  wire [0:0]                 injectdbiterra,
    output wire [0:0]                 sbiterrb,
    output wire [0:0]                 dbiterrb
);
    localparam integer DEPTH = (MEMORY_SIZE / WRITE_DATA_WIDTH_A);
    reg [WRITE_DATA_WIDTH_A-1:0] mem [0:DEPTH-1];
    integer mi;

    initial begin
        for (mi = 0; mi < DEPTH; mi = mi + 1)
            mem[mi] = {WRITE_DATA_WIDTH_A{1'b0}};
        doutb = {READ_DATA_WIDTH_B{1'b0}};
    end

    assign sbiterrb = 1'b0;
    assign dbiterrb = 1'b0;

    always @(posedge clka) begin
        if (ena && (wea != {((WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A)){1'b0}}) && !sleep) begin
            mem[addra] <= dina;
        end
    end

    always @(posedge clkb) begin
        if (rstb || sleep) begin
            doutb <= {READ_DATA_WIDTH_B{1'b0}};
        end else if (enb && regceb) begin
            doutb <= mem[addrb];
        end
    end
endmodule
