// RMSNorm, bit-exact with numerics_v2 runq_runtime._rms_high.
//
// Capture (one x/gamma pair per beat): ss += x*x in int64 (numpy wrap semantics).
// Then, sequentially with one shared seq_mul:
//   mean  = div_const_rne(ss, D) + eps          reciprocal estimate + correction (fixed.py)
//   root  = max(1, isqrt(mean << 30))           47-cycle digit-by-digit square root
//   recip = div_rne(2^54, root)                 55-cycle restoring divider + RNE fixup
// and per element (about 55 cycles each):
//   out = rshift_rne(rshift_rne(rshift_rne(x*recip, 24) * gamma, gfb) * mult, shift)
// Element products keep the low 64 bits like the numpy int64 golden.
// recip_d = round(2^32 / D) (fixed.py table). RMSNorm runs twice per layer per token, so a
// sequential datapath costs a negligible fraction of the weight-stream-bound token time.
module rmsnorm
    import decode_pkg::*;
#(
    parameter int MAX_D = 512
) (
    input  logic               clk,
    input  logic               rst,
    input  logic [32:0]        recip_d,
    input  logic [4:0]         gfb,
    input  logic [31:0]        mult,
    input  logic [6:0]         shift,
    input  logic signed [63:0] eps,
    input  logic               in_valid,
    input  logic               in_last,
    input  logic signed [31:0] x_in,
    input  logic [15:0]        gamma_in,
    output logic               in_ready,
    output logic               out_valid,
    output logic signed [31:0] out,
    output logic               done
);
    localparam int AW = $clog2(MAX_D);

    typedef enum logic [3:0] {
        CAP, SQ_FLUSH, MEAN_MUL, MEAN_Q0, MEAN_R0, MEAN_FIX, MEAN_EPS, SQRT, RECIP,
        E_LOAD, E_MUL, E_SHIFT, E_NEXT
    } state_t;
    state_t state;

    (* ram_style = "distributed" *) logic signed [31:0] xmem [MAX_D];
    (* ram_style = "distributed" *) logic [15:0]        gmem [MAX_D];
    logic [AW:0]        count, idx;
    logic signed [63:0] ss, sq;
    logic               sq_v;
    logic [63:0]        ss_mag, q0, qf;
    logic signed [79:0] r0;
    logic               ss_neg;

    // shared sequential multiplier
    logic         mul_start, mul_done;
    logic [63:0]  mul_a, mul_b;
    logic [127:0] mul_p;
    seq_mul mul (.clk, .rst, .start(mul_start), .a(mul_a), .b(mul_b), .done(mul_done), .p(mul_p));

    // isqrt state: N = mean << 30 (94 bits), 47 result bits
    logic [93:0]        sq_n;
    logic [50:0]        sq_rem;
    logic [46:0]        root;
    logic [5:0]         sq_i;
    // divider state: 2^54 / root
    logic [55:0]        dv_rem;
    logic [54:0]        recip;
    logic [5:0]         dv_i;
    // element micro-sequence: step 0: x*recip >> 24, 1: *gamma >> gfb, 2: *mult >> shift
    logic [1:0]         step;
    logic [15:0]        gamma_r;

    assign in_ready = (state == CAP);

    wire [50:0] sq_cur   = {sq_rem[48:0], sq_n[93:92]};
    wire [50:0] sq_trial = {2'b00, root, 2'b01};
    wire        sq_ge    = sq_cur >= sq_trial;
    wire [46:0] root_eff = (root == 0) ? 47'd1 : root;   // max(1, isqrt)
    wire [55:0] dv_cur   = {dv_rem[54:0], (dv_i == 6'd54)};  // numerator 2^54
    wire        dv_ge    = dv_cur >= 56'(root_eff);
    wire [55:0] dv_next  = dv_ge ? dv_cur - 56'(root_eff) : dv_cur;
    wire [6:0]  step_sh  = (step == 2'd0) ? 7'd24 : (step == 2'd1) ? 7'(gfb) : shift;
    wire signed [63:0] shifted = rshift_rne($signed(mul_p[63:0]), step_sh);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= CAP; count <= '0; ss <= '0; sq_v <= 1'b0; done <= 1'b0;
            out_valid <= 1'b0; mul_start <= 1'b0;
        end else begin
            done <= 1'b0; out_valid <= 1'b0; mul_start <= 1'b0;
            // x*x is registered before accumulation (DSP-friendly)
            sq   <= 64'(x_in) * 64'(x_in);
            sq_v <= (state == CAP) && in_valid;
            if (sq_v) ss <= ss + sq;
            case (state)
                CAP: if (in_valid) begin
                    xmem[count[AW-1:0]] <= x_in;
                    gmem[count[AW-1:0]] <= gamma_in;
                    count <= count + 1'b1;
                    if (in_last) state <= SQ_FLUSH;
                end
                SQ_FLUSH: if (!sq_v) begin   // last square has been accumulated
                    ss_neg <= ss[63];
                    ss_mag <= ss[63] ? 64'(-ss) : 64'(ss);
                    mul_a <= ss[63] ? 64'(-ss) : 64'(ss); mul_b <= 64'(recip_d); mul_start <= 1'b1;
                    state <= MEAN_MUL;
                end
                MEAN_MUL: if (mul_done) state <= MEAN_Q0;
                MEAN_Q0: begin
                    q0 <= div_const_q0_from_prod(ss_mag, mul_p[96:0]);
                    state <= MEAN_R0;
                end
                MEAN_R0: begin
                    r0 <= div_resid(ss_mag, q0, 10'(count));
                    state <= MEAN_FIX;
                end
                MEAN_FIX: begin
                    qf <= div_const_fix_resid(ss_mag > 64'h8000_0000, q0, r0, 10'(count));
                    state <= MEAN_EPS;
                end
                MEAN_EPS: begin
                    sq_n  <= {(ss_neg ? -qf : qf) + eps, 30'd0};
                    sq_rem <= '0; root <= '0; sq_i <= 6'd0;
                    state <= SQRT;
                end
                SQRT: begin
                    sq_rem <= sq_ge ? sq_cur - sq_trial : sq_cur;
                    root   <= {root[45:0], sq_ge};
                    sq_n   <= sq_n << 2;
                    sq_i   <= sq_i + 1'b1;
                    if (sq_i == 6'd46) begin
                        state <= RECIP; dv_rem <= '0; recip <= '0; dv_i <= 6'd54;
                    end
                end
                RECIP: begin
                    dv_rem <= dv_next;
                    recip  <= {recip[53:0], dv_ge};
                    dv_i   <= dv_i - 1'b1;
                    if (dv_i == 6'd0) begin
                        // div_rne: round half to even on the final remainder
                        recip <= {recip[53:0], dv_ge} + 55'(({dv_next, 1'b0} > {1'b0, 56'(root_eff)}) ||
                                 (({dv_next, 1'b0} == {1'b0, 56'(root_eff)}) && dv_ge));
                        state <= E_LOAD; idx <= '0;
                    end
                end
                E_LOAD: begin
                    mul_a <= 64'(xmem[idx[AW-1:0]]); mul_b <= 64'(recip); mul_start <= 1'b1;
                    gamma_r <= gmem[idx[AW-1:0]];
                    step <= 2'd0; state <= E_MUL;
                end
                E_MUL: if (mul_done) state <= E_SHIFT;
                E_SHIFT: begin
                    if (step == 2'd2) begin
                        out <= 32'(shifted);
`ifdef RMS_MUTATE
                        out <= 32'(shifted) ^ 32'd1;
`endif
                        out_valid <= 1'b1;
                        state <= E_NEXT;
                    end else begin
                        mul_a <= shifted;
                        mul_b <= (step == 2'd0) ? 64'(gamma_r) : 64'(mult);
                        mul_start <= 1'b1;
                        step <= step + 1'b1;
                        state <= E_MUL;
                    end
                end
                E_NEXT: begin
                    idx <= idx + 1'b1;
                    if (idx + 1'b1 == count) begin
                        state <= CAP; count <= '0; ss <= '0; done <= 1'b1;
                    end else state <= E_LOAD;
                end
                default: state <= CAP;
            endcase
        end
    end
endmodule
