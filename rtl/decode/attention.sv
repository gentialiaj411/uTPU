// Decode attention for one head, bit-exact with numerics_v2 runq_runtime (decode path):
//   score[t] = _score(q, K[t])     dot(q8, k8) * q_scale * k_scale * sc_small(score_mult)
//                                  -> rshift_rne(., score_shift-16) -> div127 -> div127 | div7
//   prob     = _softmax(score)     exp LUT on clip(score - max, -1024, 0), div_rne(32767<<24, total)
//   out      = _attention_value(prob, _to_base(V))   16-position blocks: to_base (div by qmax),
//              block peak, div_rne(127<<24, peak), INT8 codes, prob . codes, rshift_rne(.*peak, 15),
//              div127, accumulate.
// The KV cache lives here; codes are stored in CB bits (8 for KV8, 4 suffices for KV4) next to a
// 32-bit per-head scale. q arrives already quantized (act_quant over the head, as _score does).
module attention
    import decode_pkg::*;
#(
    parameter int    LAYERS   = 5,
    parameter int    MAX_SEQ  = 64,
    parameter int    KV_HEADS = 4,
    parameter int    HS       = 8,
    parameter int    CB       = 8,
    parameter int    LANES    = 8,
    parameter EXP_LUT  = "rtl/decode/luts/exp_q15.mem"
) (
    input  logic                clk,
    input  logic                rst,
    // configuration
    input  logic [31:0]         score_mult,
    input  logic [6:0]          score_shift,
    input  logic                kv4,            // 1: KV qmax 7, 0: qmax 127
    // KV cache write port
    input  logic                kv_we,
    input  logic                kv_is_v,
    input  logic [$clog2(LAYERS)-1:0]   kv_layer,
    input  logic [$clog2(MAX_SEQ)-1:0]  kv_pos,
    input  logic [$clog2(KV_HEADS)-1:0] kv_head,
    input  logic [HS*8-1:0]     kv_codes,
    input  logic [31:0]         kv_scale,
    // head operation
    input  logic                start,
    input  logic [$clog2(LAYERS)-1:0]   layer,
    input  logic [$clog2(KV_HEADS)-1:0] head,
    input  logic [$clog2(MAX_SEQ)-1:0]  pos,
    input  logic [HS*8-1:0]     q_codes,
    input  logic [31:0]         q_scale,
    output logic                busy,
    output logic                out_valid,
    output logic signed [31:0]  out,
    output logic                done
);
    localparam int DEPTH = LAYERS * MAX_SEQ * KV_HEADS;
    localparam int AW    = $clog2(DEPTH);
    localparam int TW    = $clog2(MAX_SEQ) + 1;
    localparam int W     = 32 + HS * CB;
    localparam int JW    = $clog2(HS) + 1;
    localparam int BW    = $clog2(16 * HS);

    // ---------------- KV cache ----------------
    // K entries [0, DEPTH) and V entries [DEPTH, 2*DEPTH) share one read port (scores and the AV
    // pass never overlap). The word is split into slices of up to 72 bits, each built from explicit
    // banks sized to fill a BRAM36 in its widest aspect (72x512 ... 9x4096), so Vivado does not round
    // a non-power-of-2 depth up to the next power of 2. Read: raddr -> bank output reg -> kvq_r.
    localparam int TOTD = 2 * DEPTH;
    localparam int FA   = $clog2(TOTD);
    localparam int NSL  = (W + 71) / 72;
    logic [W-1:0] wword;
    always_comb begin
        wword = '0;
        wword[W-1 -: 32] = kv_scale;
        for (int j = 0; j < HS; j++) wword[j*CB +: CB] = kv_codes[j*8 +: CB];
    end
    wire [FA-1:0] wfa = FA'((32'(kv_layer) * MAX_SEQ + 32'(kv_pos)) * KV_HEADS + 32'(kv_head) + (kv_is_v ? DEPTH : 0));
    logic [FA-1:0] raddr, raddr_d;
    logic [W-1:0]  kvq_r;
    always_ff @(posedge clk) raddr_d <= raddr;
    for (genvar sl = 0; sl < NSL; sl++) begin : g_sl
        localparam int SW  = (W - 72 * sl > 72) ? 72 : (W - 72 * sl);
        localparam int BD  = (SW <= 9) ? 4096 : (SW <= 18) ? 2048 : (SW <= 36) ? 1024 : 512;
        localparam int BAW = $clog2(BD);
        localparam int NF  = TOTD / BD;                  // full banks
        localparam int REM = TOTD % BD;                  // the last, partial bank is trimmed to a power of 2
        localparam int NB  = NF + (REM != 0 ? 1 : 0);
        logic [SW-1:0] bq [NB];
        for (genvar bk = 0; bk < NB; bk++) begin : g_b
            localparam int DB  = (bk < NF) ? BD : (1 << $clog2(REM));
            localparam int DAW = $clog2(DB);
            logic [SW-1:0] mem [DB];
            always_ff @(posedge clk) begin
                if (kv_we && (32'(wfa) >> BAW) == bk) mem[wfa[DAW-1:0]] <= wword[sl*72 +: SW];
                bq[bk] <= mem[raddr[DAW-1:0]];
            end
        end
        always_ff @(posedge clk) kvq_r[sl*72 +: SW] <= bq[32'(raddr_d) >> BAW];
    end
    function automatic logic signed [7:0] code_of(input logic [W-1:0] w, input int j);
        return 8'($signed(w[j*CB +: CB]));
    endfunction

    // ---------------- shared pipelines ----------------
    logic               mp_v, mp_ov;
    logic signed [63:0] mp_a;
    logic signed [32:0] mp_b;
    logic signed [96:0] mp_p;
    mul_pipe mp (.clk, .in_valid(mp_v), .a(mp_a), .b(mp_b), .out_valid(mp_ov), .p(mp_p));

    logic               dp_v, dp_ov, dp_d127;
    logic signed [63:0] dp_n, dp_q;
    logic [7:0]         dp_tag_in, dp_tag;
    div_const_pipe dp (.clk, .in_valid(dp_v), .n(dp_n), .d127(dp_d127), .tag_in(dp_tag_in),
                       .out_valid(dp_ov), .q(dp_q), .tag_out(dp_tag));

    (* rom_style = "distributed" *) logic [15:0] exp_rom [1025];
    initial $readmemh(EXP_LUT, exp_rom);

    // ---------------- buffers ----------------
    logic signed [31:0] sbuf [MAX_SEQ];
    logic [15:0]        ebuf [MAX_SEQ];
    logic [15:0]        pbuf [16];
    logic signed [32:0] vbuf [16 * HS];
    logic signed [31:0] abuf [HS];
    logic signed [63:0] raw  [HS];

    typedef enum logic [5:0] {
        IDLE, SC_RD, SC_WAIT, SC_WAIT2, SC_DOT, SC_SUM, SC_MUL, SC_MWAIT, SC_SH, SC_DIV, SC_DWAIT, SC_STORE,
        SM_RD, SM_IDX, SM_ACC, SM_DIV, AV_PRD, AV_PMUL, AV_VRD, AV_VWAIT, AV_VWAIT2, AV_VISSUE, AV_VDRAIN,
        AV_RDIV, AV_ARD, AV_AMUL, AV_ACODE, AV_AACC, AV_FMUL, AV_FWAIT, AV_FDIV, AV_FDWAIT, OUT
    } state_t;
    state_t state;

    logic [$clog2(LAYERS)-1:0]   L;
    logic [$clog2(MAX_SEQ)-1:0]  P;
    logic signed [7:0]  qc [HS];
    logic [31:0]        qs;
    logic [AW-1:0]      base_addr;
    logic [TW-1:0]      t, blk, bw;           // position, block start, block width
    logic [JW-1:0]      j;
    logic [4:0]         i;
    logic [1:0]         step;
    logic signed [31:0] acc, smax;
    logic signed [63:0] p;
    logic [31:0]        kscale, total;
    logic [5:0]         div_i;
    logic [40:0]        dv_rem;
    logic [39:0]        dv_num, dv_q;
    logic [32:0]        dv_den;
    logic [39:0]        precip;
    logic [31:0]        vrecip;
    logic [32:0]        peak;
    logic [BW:0]        n_issue, n_recv;
    logic signed [31:0] vscale;
    logic signed [7:0]  vcode [HS];
    logic signed [65:0] m65;
    logic signed [15:0] prods [LANES];
    logic signed [32:0] vtmp;
    logic signed [31:0] code_r;

    wire [63:0] sc_small = 64'(rshift_rne(64'(score_mult), 7'd16));
    assign busy = (state != IDLE);

    // restoring divider step (shared by the softmax and block reciprocals)
    wire [40:0] dv_cur  = {dv_rem[39:0], dv_num[39]};
    wire        dv_ge   = dv_cur >= 41'(dv_den);
    wire [40:0] dv_next = dv_ge ? dv_cur - 41'(dv_den) : dv_cur;
    wire [39:0] dv_qn   = {dv_q[38:0], dv_ge};
    wire        dv_up   = ({dv_next, 1'b0} > {1'b0, 41'(dv_den)}) || (({dv_next, 1'b0} == {1'b0, 41'(dv_den)}) && dv_ge);

    always_ff @(posedge clk) begin
        mp_v <= 1'b0; dp_v <= 1'b0; out_valid <= 1'b0; done <= 1'b0;
        if (rst) begin
            state <= IDLE;
        end else case (state)
            IDLE: if (start) begin
                L <= layer; P <= pos; qs <= q_scale;
                for (int k = 0; k < HS; k++) qc[k] <= q_codes[k*8 +: 8];
                base_addr <= AW'(32'(layer) * MAX_SEQ * KV_HEADS + 32'(head));
                t <= '0; smax <= 32'sh8000_0000;
                for (int k = 0; k < HS; k++) raw[k] <= '0;
                state <= SC_RD;
            end
            // ---------------- scores ----------------
            SC_RD: begin raddr <= FA'(base_addr + AW'(32'(t) * KV_HEADS)); state <= SC_WAIT; end
            SC_WAIT: begin j <= '0; acc <= '0; state <= SC_WAIT2; end   // raddr -> kq
            SC_WAIT2: state <= SC_DOT;                                    // kq -> kq_r
            SC_DOT: begin
                for (int l = 0; l < LANES; l++) prods[l] <= 16'(qc[j + l]) * 16'(code_of(kvq_r, int'(j) + l));
                state <= SC_SUM;
            end
            SC_SUM: begin
                logic signed [31:0] s;
                s = '0;
                for (int l = 0; l < LANES; l++) s = s + 32'(prods[l]);
                acc <= acc + s;
                j <= j + JW'(LANES);
                if (j + JW'(LANES) == JW'(HS)) begin kscale <= kvq_r[W-1 -: 32]; step <= 2'd0; state <= SC_MUL; end
                else state <= SC_DOT;
            end
            SC_MUL: begin
                mp_v <= 1'b1;
                case (step)
                    2'd0: begin mp_a <= 64'(acc); mp_b <= {1'b0, qs}; end
                    2'd1: begin mp_a <= p; mp_b <= {1'b0, kscale}; end
                    default: begin mp_a <= p; mp_b <= 33'(sc_small); end
                endcase
                state <= SC_MWAIT;
            end
            SC_MWAIT: if (mp_ov) begin
                p <= mp_p[63:0];
                step <= step + 1'b1;
                state <= (step == 2'd2) ? SC_SH : SC_MUL;
            end
            SC_SH: begin
                dp_v <= 1'b1; dp_n <= rshift_rne(p, score_shift - 7'd16); dp_d127 <= 1'b1;
                step <= 2'd0; state <= SC_DWAIT;
            end
            SC_DIV: begin dp_v <= 1'b1; dp_n <= p; dp_d127 <= !kv4; state <= SC_DWAIT; end
            SC_DWAIT: if (dp_ov) begin
                p <= dp_q;
                step <= step + 1'b1;
                state <= (step == 2'd0) ? SC_DIV : SC_STORE;
            end
            SC_STORE: begin
                sbuf[t[TW-2:0]] <= p[31:0];
                if ($signed(p[31:0]) > smax) smax <= p[31:0];
                if (t == TW'(P)) begin t <= '0; total <= '0; state <= SM_RD; end
                else begin t <= t + 1'b1; state <= SC_RD; end
            end
            // ---------------- softmax ----------------
            SM_RD: begin acc <= sbuf[t[TW-2:0]]; state <= SM_IDX; end
            SM_IDX: begin
                logic signed [33:0] d;
                d = 34'(smax) - 34'(acc);                              // >= 0
                kscale <= 32'(exp_rom[(d > 34'sd1024) ? 11'd1024 : 11'(d)]);
                state <= SM_ACC;
            end
            SM_ACC: begin
                ebuf[t[TW-2:0]] <= kscale[15:0];
                total <= total + kscale;
                if (t == TW'(P)) begin
                    dv_num <= 40'(32767) << 24; dv_den <= 33'(total + kscale);
                    dv_rem <= '0; dv_q <= '0; div_i <= 6'd39; state <= SM_DIV;
                end else begin t <= t + 1'b1; state <= SM_RD; end
            end
            SM_DIV: begin
                dv_rem <= dv_next; dv_q <= dv_qn; dv_num <= dv_num << 1; div_i <= div_i - 1'b1;
                if (div_i == 6'd0) begin
                    precip <= dv_qn + 40'(dv_up);
                    blk <= '0; state <= AV_PRD; i <= '0;
                end
            end
            // ---------------- attention-value blocks ----------------
            AV_PRD: begin
                bw <= ((TW'(P) + 1'b1 - blk) > TW'(16)) ? TW'(16) : (TW'(P) + 1'b1 - blk);
                kscale <= 32'(ebuf[(blk + TW'(i))]);
                state <= AV_PMUL;
            end
            AV_PMUL: begin
                pbuf[i[3:0]] <= 16'(rshift_rne(64'(kscale) * 64'(precip), 7'd24));
                if (TW'(i) + 1'b1 == bw) begin
                    i <= '0; peak <= 33'd1; n_issue <= '0; n_recv <= '0; state <= AV_VRD;
                end else begin i <= i + 1'b1; state <= AV_PRD; end
            end
            AV_VRD: begin raddr <= FA'(32'(base_addr + AW'(32'(blk + TW'(i)) * KV_HEADS)) + DEPTH); state <= AV_VWAIT; end
            AV_VWAIT: begin j <= '0; state <= AV_VWAIT2; end
            AV_VWAIT2: state <= AV_VISSUE;
            AV_VISSUE: begin
                // _to_base: div_const_rne(code * scale, qmax), one element per cycle
                dp_v <= 1'b1; dp_d127 <= !kv4;
                dp_n <= 64'(code_of(kvq_r, int'(j))) * 64'($signed(kvq_r[W-1 -: 32]));
                dp_tag_in <= 8'(n_issue);
                n_issue <= n_issue + 1'b1;
                j <= j + 1'b1;
                if (j == JW'(HS - 1)) begin
                    if (TW'(i) + 1'b1 == bw) state <= AV_VDRAIN;
                    else begin i <= i + 1'b1; state <= AV_VRD; end
                end
            end
            AV_VDRAIN: if (n_recv == (BW+1)'(32'(bw) * HS)) begin
                dv_num <= 40'(127) << 24; dv_den <= peak; dv_rem <= '0; dv_q <= '0; div_i <= 6'd39;
                state <= AV_RDIV;
            end
            AV_RDIV: begin
                dv_rem <= dv_next; dv_q <= dv_qn; dv_num <= dv_num << 1; div_i <= div_i - 1'b1;
                if (div_i == 6'd0) begin
                    vrecip <= 32'(dv_qn + 40'(dv_up));
                    j <= '0; i <= '0; acc <= '0; state <= AV_ARD;
                end
            end
            AV_ARD: begin vtmp <= vbuf[32'(i) * HS + 32'(j)]; state <= AV_AMUL; end
            AV_AMUL: begin m65 <= 66'(vtmp) * $signed({34'd0, vrecip}); state <= AV_ACODE; end
            AV_ACODE: begin
                logic signed [63:0] r;
                r = rshift_rne(m65[63:0], 7'd24);
                code_r <= (r > 64'sd127) ? 32'sd127 : (r < -64'sd128) ? -32'sd128 : 32'(r);
                state <= AV_AACC;
            end
            AV_AACC: begin
                logic signed [31:0] c;
                c = code_r;
                if (TW'(i) + 1'b1 == bw) begin
                    abuf[j[JW-2:0]] <= acc + $signed({16'd0, pbuf[i[3:0]]}) * c;
                    acc <= '0; i <= '0;
                    if (j == JW'(HS - 1)) begin j <= '0; state <= AV_FMUL; end
                    else begin j <= j + 1'b1; state <= AV_ARD; end
                end else begin
                    acc <= acc + $signed({16'd0, pbuf[i[3:0]]}) * c;
                    i <= i + 1'b1; state <= AV_ARD;
                end
            end
            AV_FMUL: begin mp_v <= 1'b1; mp_a <= 64'(abuf[j[JW-2:0]]); mp_b <= peak; state <= AV_FWAIT; end
            AV_FWAIT: if (mp_ov) begin
                dp_v <= 1'b1; dp_d127 <= 1'b1; dp_n <= rshift_rne(mp_p[63:0], 7'd15); state <= AV_FDWAIT;
            end
            AV_FDWAIT: if (dp_ov) begin
                raw[j[JW-2:0]] <= raw[j[JW-2:0]] + dp_q;
                if (j == JW'(HS - 1)) begin
                    j <= '0;
                    if (blk + bw > TW'(P)) state <= OUT;
                    else begin blk <= blk + TW'(16); i <= '0; state <= AV_PRD; end
                end else begin j <= j + 1'b1; state <= AV_FMUL; end
            end
            OUT: begin
                out <= raw[j[JW-2:0]][31:0];
`ifdef ATTN_MUTATE
                out <= raw[j[JW-2:0]][31:0] ^ 32'd1;
`endif
                out_valid <= 1'b1;
                j <= j + 1'b1;
                if (j == JW'(HS - 1)) begin done <= 1'b1; state <= IDLE; end
            end
            default: state <= IDLE;
        endcase
        // _to_base results come back in issue order
        if (state == AV_VRD || state == AV_VWAIT || state == AV_VWAIT2 || state == AV_VISSUE || state == AV_VDRAIN) begin
            if (dp_ov) begin
                logic [32:0] mag;
                vbuf[n_recv[BW-1:0]] <= 33'(dp_q);
                mag = dp_q[63] ? 33'(-dp_q) : 33'(dp_q);
                if (mag > peak) peak <= mag;
                n_recv <= n_recv + 1'b1;
            end
        end
    end
endmodule
