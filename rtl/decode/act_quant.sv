// Dynamic group activation quantizer, bit-exact with numerics_v2 runq_runtime._quantize (qmax=127).
//
// Per group (<= 64 INT32 values, in_last on the final one):
//   peak  = max(1, max |x|)                         tracked while capturing
//   recip = div_rne(127 << 24, peak)                31-cycle restoring divider + RNE fixup
//   code  = clamp(rshift_rne(x * recip, 24), +-127) 2-stage pipeline, one code per cycle
// scale_out is the peak (QValue.scale_multiplier), valid with done.
module act_quant #(
    parameter int MAX_GROUP = 64
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               in_valid,
    input  logic               in_last,
    input  logic signed [31:0] raw_in,
    output logic               in_ready,
    output logic               code_valid,
    output logic signed [7:0]  code_out,
    output logic [31:0]        scale_out,
    output logic               done
);
    localparam int AW = $clog2(MAX_GROUP);
    localparam logic [30:0] NUMER = 31'(127) << 24;

    typedef enum logic [1:0] {CAP, DIV, EMIT, DRAIN} state_t;
    state_t state;

    logic signed [31:0] mem [MAX_GROUP];
    logic [AW:0]        count, emit_i;
    logic [31:0]        peak;
    logic [4:0]         div_bit;
    logic [31:0]        rem, quot;
    logic [31:0]        recip;

    assign in_ready = (state == CAP);

    wire [31:0] mag_in = raw_in[31] ? 32'(-raw_in) : 32'(raw_in);
    // restoring step: shift in the next numerator bit
    wire [32:0] rem_shift = {rem, NUMER[div_bit]};
    wire        rem_ge    = rem_shift >= {1'b0, peak};
    wire [31:0] rem_next  = rem_ge ? 32'(rem_shift - {1'b0, peak}) : rem_shift[31:0];
    wire [31:0] quot_next = {quot[30:0], rem_ge};

    // emit pipeline
    logic               v1, v2, last1, last2;
    logic signed [63:0] prod1;
    logic signed [8:0]  code2;

    function automatic logic signed [8:0] round_clamp(input logic signed [63:0] p);
        logic [63:0] mag, q;
        logic [23:0] rem24;
        logic        inc;
        mag   = p[63] ? 64'(-p) : 64'(p);
        q     = mag >> 24;
        rem24 = mag[23:0];
        inc   = (rem24 > 24'h800000) || ((rem24 == 24'h800000) && q[0]);
        q     = q + 64'(inc);
        if (q > 64'd127) q = 64'd127;
        return p[63] ? -$signed({1'b0, q[7:0]}) : $signed({1'b0, q[7:0]});
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= CAP; count <= '0; peak <= 32'd1; done <= 1'b0;
            v1 <= 1'b0; v2 <= 1'b0; code_valid <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                CAP: if (in_valid) begin
                    mem[count[AW-1:0]] <= raw_in;
                    count <= count + 1'b1;
                    if (mag_in > peak) peak <= mag_in;
                    if (in_last) begin
                        state <= DIV; div_bit <= 5'd30; rem <= '0; quot <= '0;
                        scale_out <= (mag_in > peak) ? mag_in : peak;
                    end
                end
                DIV: begin
                    // peak is final here (last capture already compared)
                    rem <= rem_next; quot <= quot_next;
                    div_bit <= div_bit - 1'b1;
                    if (div_bit == 5'd0) begin
                        // div_rne: round half to even on the remainder
                        recip <= quot_next + 32'(({rem_next, 1'b0} > {1'b0, peak}) ||
                                                 (({rem_next, 1'b0} == {1'b0, peak}) && quot_next[0]));
                        state <= EMIT; emit_i <= '0;
                    end
                end
                EMIT: begin
                    emit_i <= emit_i + 1'b1;
                    if (emit_i + 1'b1 == count) state <= DRAIN;
                end
                DRAIN: if (!v1 && !v2) begin
                    state <= CAP; count <= '0; peak <= 32'd1; done <= 1'b1;
                end
            endcase

            // stage 1: multiply
            v1    <= (state == EMIT);
            last1 <= (state == EMIT) && (emit_i + 1'b1 == count);
            prod1 <= 64'(mem[emit_i[AW-1:0]]) * $signed({32'd0, recip});
            // stage 2: round + clamp
            v2    <= v1; last2 <= last1;
            code2 <= round_clamp(prod1);
            code_valid <= v2;
            code_out   <= code2[7:0];
`ifdef ACTQ_MUTATE
            if (v2 && mutate_idx == 3'd5) code_out <= code2[7:0] ^ 8'd1;
`endif
        end
    end

`ifdef ACTQ_MUTATE
    logic [2:0] mutate_idx;   // counts codes within a group up to 5
    always_ff @(posedge clk)
        if (rst || done) mutate_idx <= '0;
        else if (v2 && mutate_idx != 3'd7) mutate_idx <= mutate_idx + 1'b1;
`endif
endmodule
