// Sequential unsigned 64x64 -> 128 multiplier, radix 16 (one 64x4 partial product per cycle).
// 16 cycles start-to-done; no DSPs. Callers pass two's-complement operands when they only
// need the low 64 bits (numpy int64 wrap), or magnitudes when they need the full product.
module seq_mul (
    input  logic          clk,
    input  logic          rst,
    input  logic          start,
    input  logic [63:0]   a,
    input  logic [63:0]   b,
    output logic          done,
    output logic [127:0]  p
);
    logic [63:0]  a_r, b_r;
    logic [131:0] acc;
    logic [4:0]   i;
    logic         busy;

    wire [67:0] sum = {4'd0, acc[131:68]} + a_r * b_r[3:0];

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin
                a_r <= a; b_r <= b; acc <= '0; i <= '0; busy <= 1'b1;
            end else if (busy) begin
                acc <= {sum, acc[67:4]};
                b_r <= b_r >> 4;
                i <= i + 1'b1;
                if (i == 5'd15) begin busy <= 1'b0; done <= 1'b1; end
            end
        end
    end
    // after 16 steps the product sits in acc[131:4]
    assign p = acc[131:4];
endmodule
