// Element-wise decode ops, bit-exact with numerics_v2 runq_runtime:
//   OP_ROPE     (a,b) = pair (x0,x1)  -> _rope_high: rshift_rne(x0*c - x1*s, 15), rshift_rne(x0*s + x1*c, 15)
//                 c/s from the RoPE ROM at rope_base + pair index (pair index wraps at rope_half;
//                 op_start resets it). rope_base = position * rope_half.
//   OP_SWIGLU   (a,b) = (h1,h3)       -> _hidden_high(_silu_high(h1), h3)
//   OP_ADD      (a,b)                 -> int32 wrap-around a + b (residual)
//   OP_MULSHIFT (a,m,sh)              -> rshift_rne(a*m, sh) (tied embedding row scaling)
// One shared 33x33 multiplier with registered operands and product (3 cycles per product).
// Results are the low 32 bits, matching the golden's astype(np.int32).
module vec_alu
    import decode_pkg::*;
#(
    parameter int    ROPE_DEPTH = 2048,
    parameter string ROPE_COS   = "rtl/decode/luts/rope_cos_q15_h8.mem",
    parameter string ROPE_SIN   = "rtl/decode/luts/rope_sin_q15_h8.mem",
    parameter string SIGMOID    = "rtl/decode/luts/sigmoid_q15.mem"
) (
    input  logic               clk,
    input  logic               rst,
    input  logic [1:0]         op,
    input  logic               op_start,
    input  logic [15:0]        rope_base,
    input  logic [5:0]         rope_half,
    input  logic [31:0]        silu_mult,
    input  logic [6:0]         silu_shift,
    input  logic [31:0]        swiglu_mult,
    input  logic [6:0]         swiglu_shift,
    input  logic               in_valid,
    input  logic signed [31:0] a,
    input  logic signed [31:0] b,
    input  logic [31:0]        m,
    input  logic [6:0]         sh,
    output logic               in_ready,
    output logic               out_valid,
    output logic signed [31:0] out0,
    output logic signed [31:0] out1
);
    localparam logic [1:0] OP_ROPE = 2'd0, OP_SWIGLU = 2'd1, OP_ADD = 2'd2, OP_MULSHIFT = 2'd3;
    localparam int RW = $clog2(ROPE_DEPTH);

    logic signed [15:0] cos_rom [ROPE_DEPTH];
    logic signed [15:0] sin_rom [ROPE_DEPTH];
    logic [15:0]        sig_rom [1025];
    initial begin
        $readmemh(ROPE_COS, cos_rom);
        $readmemh(ROPE_SIN, sin_rom);
        $readmemh(SIGMOID, sig_rom);
    end

    typedef enum logic [2:0] {IDLE, ISSUE, WAIT, TAKE, SIG_READ, SIG_SEL, OUT} state_t;
    state_t state;

    logic signed [31:0] a_r, b_r;
    logic [31:0]        m_r;
    logic [6:0]         sh_r;
    logic [1:0]         op_r;
    logic [2:0]         k;
    logic [5:0]         pair_i;
    logic signed [15:0] c_r, s_r;
    logic signed [63:0] t0, t1, t2, t3, idx_r;
    logic [15:0]        sig_q, sig_r;

    // shared multiplier
    logic signed [32:0] ma, mb;
    logic signed [65:0] prod;
    always_ff @(posedge clk) prod <= ma * mb;
    wire signed [63:0] p = prod[63:0];

    wire [63:0] sw_small = 64'(rshift_rne(64'(swiglu_mult), 7'd16));

    // operands for multiply k of the current op
    logic signed [32:0] opa, opb;
    always_comb begin
        opa = 33'(a_r); opb = 33'(b_r);
        case (op_r)
            OP_ROPE: case (k)
                3'd0: begin opa = 33'(a_r); opb = 33'(c_r); end
                3'd1: begin opa = 33'(b_r); opb = 33'(s_r); end
                3'd2: begin opa = 33'(a_r); opb = 33'(s_r); end
                default: begin opa = 33'(b_r); opb = 33'(c_r); end
            endcase
            OP_SWIGLU: case (k)
                3'd0: begin opa = 33'(a_r); opb = {1'b0, silu_mult}; end
                3'd1: begin opa = 33'(a_r); opb = {17'd0, sig_r}; end
                3'd2: begin opa = 33'($signed(t0[31:0])); opb = 33'(b_r); end
                3'd3: begin opa = {1'b0, t1[31:0]}; opb = 33'(sw_small); end
                default: begin opa = 33'($signed(t1[63:32])); opb = 33'(sw_small); end
            endcase
            default: begin opa = 33'(a_r); opb = {1'b0, m_r}; end   // OP_MULSHIFT
        endcase
    end

    wire [2:0] last_k = (op_r == OP_ROPE) ? 3'd3 : (op_r == OP_SWIGLU) ? 3'd4 : 3'd0;
    wire signed [63:0] low64 = t2 + (t3 <<< 32);

    assign in_ready = (state == IDLE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; out_valid <= 1'b0; pair_i <= '0;
        end else begin
            out_valid <= 1'b0;
            if (op_start) pair_i <= '0;
            case (state)
                IDLE: if (in_valid) begin
                    a_r <= a; b_r <= b; m_r <= m; sh_r <= sh; op_r <= op; k <= '0;
                    c_r <= cos_rom[RW'(rope_base + 16'(pair_i))];
                    s_r <= sin_rom[RW'(rope_base + 16'(pair_i))];
                    if (op == OP_ROPE) pair_i <= (pair_i + 1'b1 == rope_half) ? '0 : pair_i + 1'b1;
                    state <= (op == OP_ADD) ? OUT : ISSUE;
                end
                ISSUE: begin ma <= opa; mb <= opb; state <= WAIT; end
                WAIT: state <= TAKE;
                TAKE: begin
                    state <= (k == last_k) ? OUT : ISSUE;
                    k <= k + 1'b1;
                    case (op_r)
                        OP_ROPE: case (k)
                            3'd0: t0 <= p;
                            3'd1: t1 <= p;
                            3'd2: t2 <= p;
                            default: t3 <= p;
                        endcase
                        OP_SWIGLU: case (k)
                            3'd0: begin idx_r <= rshift_rne(p, silu_shift); state <= SIG_READ; end
                            3'd1: t0 <= 64'($signed(32'(rshift_rne(p, 7'd15))));   // silu, int32
                            3'd2: t1 <= p;                                           // silu*h3 (exact)
                            3'd3: t2 <= p;                                           // low word * sw_small
                            default: t3 <= p;                                        // high word * sw_small
                        endcase
                        default: t0 <= p;
                    endcase
                end
                SIG_READ: begin
                    sig_q <= sig_rom[(idx_r < -64'sd512 || idx_r > 64'sd512) ? 11'd0 : 11'(idx_r + 64'sd512)];
                    state <= SIG_SEL;
                end
                SIG_SEL: begin
                    sig_r <= (idx_r < -64'sd512) ? 16'd0 : (idx_r > 64'sd512) ? 16'd32767 : sig_q;
                    state <= ISSUE;
                end
                OUT: begin
                    logic signed [31:0] r0;
                    case (op_r)
                        OP_ROPE: begin
                            r0 = 32'(rshift_rne(t0 - t1, 7'd15));
                            out1 <= 32'(rshift_rne(t2 + t3, 7'd15));
                        end
                        OP_SWIGLU: r0 = 32'(rshift_rne(low64, swiglu_shift - 7'd16));
                        OP_ADD: r0 = a_r + b_r;
                        default: r0 = 32'(rshift_rne(t0, sh_r));
                    endcase
`ifdef VEC_MUTATE
                    r0[0] = ~r0[0];
`endif
                    out0 <= r0;
                    out_valid <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
