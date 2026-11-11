// Pipelined signed 64x33 multiplier, full 97-bit product; one input per cycle, latency 4.
// Callers that model numpy int64 arithmetic keep p[63:0] (wrap-around); the low 64 bits of
// a product do not depend on any operand bits above 64.
module mul_pipe (
    input  logic               clk,
    input  logic               in_valid,
    input  logic signed [63:0] a,
    input  logic signed [32:0] b,
    output logic               out_valid,
    output logic signed [96:0] p
);
    logic [3:0]         v;
    logic signed [63:0] a_r;
    logic signed [32:0] b_r;
    logic signed [96:0] p1, p2;
    always_ff @(posedge clk) begin
        v   <= {v[2:0], in_valid};
        a_r <= a; b_r <= b;
        p1  <= a_r * b_r;
        p2  <= p1;
        p   <= p2;
    end
    assign out_valid = v[3];
endmodule
