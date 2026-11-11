// Pipelined fixed.div_const_rne(n, d) for d in {7, 127}; one input per cycle, latency 7.
// q0 from the Q32 reciprocal (RNE on the small |n| <= 2^31 path, half-up on the wide path),
// residual r0 = |n| - q0*d by shift-and-subtract, then the fixed.py correction.
module div_const_pipe
    import decode_pkg::*;
(
    input  logic               clk,
    input  logic               in_valid,
    input  logic signed [63:0] n,
    input  logic               d127,      // 1: d = 127, 0: d = 7
    input  logic [7:0]         tag_in,
    output logic               out_valid,
    output logic signed [63:0] q,
    output logic [7:0]         tag_out
);
    localparam logic [29:0] R7 = 30'd613566757, R127 = 30'd33818641;
    localparam int L = 7;

    logic [L-1:0] v;
    logic [7:0]   tag [L];
    logic         neg [L], wide [L], big [L];
    logic [63:0]  mag1, mag2, mag3, mag4, mag5;
    logic [29:0]  r1;
    logic [93:0]  prod2, prod3;
    logic [63:0]  q4, q5, q6;
    logic signed [79:0] res5;

    always_ff @(posedge clk) begin
        v <= {v[L-2:0], in_valid};
        tag[0] <= tag_in;
        for (int i = 1; i < L; i++) begin tag[i] <= tag[i-1]; neg[i] <= neg[i-1]; big[i] <= big[i-1]; end
        // 1: magnitude and path select
        neg[0] <= n[63];
        big[0] <= d127;
        mag1   <= n[63] ? 64'(-n) : 64'(n);
        r1     <= d127 ? R127 : R7;
        // 2-3: reciprocal product (two registers for DSP pipelining)
        mag2 <= mag1; prod2 <= 94'(mag1) * 94'(r1);
        mag3 <= mag2; prod3 <= prod2;
        // 4: quotient estimate
        mag4 <= mag3;
        if (mag3 <= 64'h8000_0000) begin
            logic [63:0] qq;
            qq = 64'(prod3[93:32]);
            if (prod3[31:0] > 32'h8000_0000 || (prod3[31:0] == 32'h8000_0000 && qq[0])) qq = qq + 1;
            q4 <= qq;
        end else q4 <= 64'((prod3 + (94'd1 << 31)) >> 32);
        // 5: residual mag - q0*d via shifts (d = 2^k - 1)
        mag5 <= mag4; q5 <= q4;
        res5 <= $signed({16'd0, mag4}) - ($signed({16'd0, q4}) <<< (big[3] ? 7 : 3)) + $signed({16'd0, q4});
        // 6: correction
        q6 <= div_const_fix_resid(mag5 > 64'h8000_0000, q5, res5, big[4] ? 10'd127 : 10'd7);
        // 7: sign
        q <= neg[5] ? -$signed(q6) : $signed(q6);
    end
    assign out_valid = v[L-1];
    assign tag_out   = tag[L-1];
endmodule
