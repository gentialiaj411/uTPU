// Single-token decode engine: one RunQIntegerLlama.forward_argmax step per start pulse.
//
// Weights stream from a byte image (tools/decode/pack_decode_image.py) through an 8-bit
// valid/ready port with a seek request; the KV cache and all activations stay on chip.
// A prefetcher streams each token's four image segments in order (embedding codes, embedding
// scales, layers, classifier) into a FIFO_DEPTH-byte FIFO, so flash reads overlap compute; the
// sequencer only pops bytes. Flash protocol: seek_req (one cycle) loads seek_addr; st_valid must
// be low in that cycle; afterwards bytes arrive in address order under valid/ready.
// A fixed-function sequencer walks the per-layer step list below (parameterized by the model
// configuration), driving the verified units: gemv_unit, act_quant, rmsnorm, vec_alu, attention.
//
//   embed -> seek layers -> per layer: rms_att, quant, wq, wk, wv, rope q, rope k, kv store,
//   attention, quant, wo, add, rms_ffn, quant, w1, w3, swiglu, quant, w2, add
//   -> rms_final, quant, seek classifier, classifier GEMV + argmax (lowest index wins ties).
// Scratch buffers: X residual, XB norm/attention out, Q, K, V, H1, H3, T (GEMV temp).
module top_decode
    import decode_pkg::*;
#(
    parameter int D = 64, H = 172, NL = 5, NH = 8, NKV = 4, HS = 8, VOCAB = 512, MAX_SEQ = 64,
    parameter int KV4 = 0,
    parameter int CLS4 = 1, WQ4 = 0, WK4 = 0, WV4 = 0, WO4 = 0, W14 = 1, W34 = 1, W24 = 0,
    parameter longint CLS_BASE = 0, EMB_SCALE_BASE = 21504, LAYERS_BASE = 26624, IMAGE_BYTES = 237632,
    parameter int FIFO_DEPTH = 4096,
    parameter int CLS_TILE_BYTES = 672, CLS_KB = 32, CLS_G = 2,
    parameter longint RECIP_D = 67108864, NORM_MULT = 1072658993, EPS = 297,
    parameter int GFB = 12, NORM_SHIFT = 33,
    parameter longint SILU_MULT = 930645194, SWIGLU_MULT = 888647850, SCORE_MULT = 1028527294,
    parameter int SILU_SHIFT = 37, SWIGLU_SHIFT = 43, SCORE_SHIFT = 46,
    parameter ROPE_COS = "rtl/decode/luts/rope_cos_q15_h8.mem",
    parameter ROPE_SIN = "rtl/decode/luts/rope_sin_q15_h8.mem",
    parameter int ROPE_DEPTH = 2048
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               start,
    input  logic [15:0]        token,
    input  logic [$clog2(MAX_SEQ)-1:0] pos,
    output logic               seek_req,
    output logic [31:0]        seek_addr,
    input  logic               st_valid,
    input  logic [7:0]         st_data,
    output logic               st_ready,
    output logic               busy,
    output logic               done,
    output logic [15:0]        token_out,
    output logic               logit_valid,
    output logic [15:0]        logit_idx,
    output logic signed [31:0] logit_val
);
    localparam int KVD    = NKV * HS;
    localparam int MAXLEN = (D > H) ? D : H;
    localparam int NBUF   = 8;
    localparam int SAW    = $clog2(NBUF * MAXLEN);
    localparam int LW     = $clog2(MAXLEN) + 1;
    localparam int GMAX   = (MAXLEN + 31) / 32;
    localparam int KBMAX  = MAXLEN;
    localparam int KV_MUL = NH / NKV;
    localparam int PW     = $clog2(MAX_SEQ);
    localparam logic [2:0] BX = 3'd0, BXB = 3'd1, BQ = 3'd2, BK = 3'd3, BV = 3'd4, BH1 = 3'd5, BH3 = 3'd6, BT = 3'd7;

    // ---------------- scratch memory ----------------
    logic signed [31:0] scr [NBUF * MAXLEN];
    logic [SAW-1:0]     rd_addr, wr_addr;
    logic signed [31:0] scr_q, wr_data;
    logic               wr_en;
    always_ff @(posedge clk) begin
        scr_q <= scr[rd_addr];
        if (wr_en) scr[wr_addr] <= wr_data;
    end
    function automatic logic [SAW-1:0] sa(input logic [2:0] b, input logic [LW-1:0] i);
        return SAW'(32'(b) * MAXLEN + 32'(i));
    endfunction

    logic signed [7:0]  qc [MAXLEN];
    logic signed [7:0]  qc_q;
    logic [LW-1:0]      qc_raddr, qc_ra;
    logic [31:0]        qs [GMAX];
    always_ff @(posedge clk) qc_q <= qc[qc_ra];

    // ---------------- GEMV tile buffers ----------------
    logic [7:0]  tbuf [16][KBMAX];
    logic [7:0]  tb_q [16];
    logic [LW-1:0] tb_raddr, tb_ra;
    always_ff @(posedge clk) for (int r = 0; r < 16; r++) tb_q[r] <= tbuf[r][tb_ra];
    logic [31:0] mrf [GMAX][16];
    logic [7:0]  srf [GMAX][16];

    // ---------------- units ----------------
    logic gv_start, gv_in_valid, gv_in_ready, gv_busy, gv_done, gv_int4, gv_nib, gv_chunk_last, gv_tile_last;
    logic [16*8-1:0]  gv_w;
    logic [16*32-1:0] gv_mult, gv_res;
    logic [16*8-1:0]  gv_shift;
    logic [31:0]      gv_act_scale;
    decode_gemv_unit gemv (.clk, .rst, .start(gv_start), .in_valid(gv_in_valid), .in_ready(gv_in_ready),
        .int4_mode(gv_int4), .nibble_hi(gv_nib), .act_code(qc_q), .weight_bytes(gv_w),
        .chunk_last(gv_chunk_last), .tile_last(gv_tile_last), .act_scale(gv_act_scale),
        .multipliers(gv_mult), .shifts(gv_shift), .busy(gv_busy), .done(gv_done), .result(gv_res));

    logic aq_in_valid, aq_in_last, aq_qmax7, aq_in_ready, aq_code_valid, aq_done;
    logic signed [31:0] aq_raw;
    logic signed [7:0]  aq_code;
    logic [31:0]        aq_scale;
    act_quant #(.MAX_GROUP(64)) aq (.clk, .rst, .in_valid(aq_in_valid), .in_last(aq_in_last), .qmax7(aq_qmax7),
        .raw_in(aq_raw), .in_ready(aq_in_ready), .code_valid(aq_code_valid), .code_out(aq_code),
        .scale_out(aq_scale), .done(aq_done));

    logic rm_in_valid, rm_in_last, rm_in_ready, rm_out_valid, rm_done;
    logic signed [31:0] rm_x, rm_out;
    logic [15:0] rm_gamma;
    rmsnorm #(.MAX_D(D)) rms (.clk, .rst, .recip_d(33'(RECIP_D)), .gfb(5'(GFB)), .mult(32'(NORM_MULT)),
        .shift(7'(NORM_SHIFT)), .eps(64'(EPS)), .in_valid(rm_in_valid), .in_last(rm_in_last), .x_in(rm_x),
        .gamma_in(rm_gamma), .in_ready(rm_in_ready), .out_valid(rm_out_valid), .out(rm_out), .done(rm_done));

    logic [1:0] va_op;
    logic va_op_start, va_in_valid, va_in_ready, va_out_valid;
    logic signed [31:0] va_a, va_b, va_out0, va_out1;
    logic [31:0] va_m;
    logic [6:0]  va_sh;
    vec_alu #(.ROPE_DEPTH(ROPE_DEPTH), .ROPE_COS(ROPE_COS), .ROPE_SIN(ROPE_SIN)) va (.clk, .rst, .op(va_op),
        .op_start(va_op_start), .rope_base(16'(32'(pos_r) * (HS / 2))), .rope_half(6'(HS / 2)),
        .silu_mult(32'(SILU_MULT)), .silu_shift(7'(SILU_SHIFT)), .swiglu_mult(32'(SWIGLU_MULT)),
        .swiglu_shift(7'(SWIGLU_SHIFT)), .in_valid(va_in_valid), .a(va_a), .b(va_b), .m(va_m), .sh(va_sh),
        .in_ready(va_in_ready), .out_valid(va_out_valid), .out0(va_out0), .out1(va_out1));

    logic at_kv_we, at_kv_is_v, at_start, at_busy, at_out_valid, at_done;
    logic [HS*8-1:0] hcodes;
    logic [31:0]     hscale;
    logic signed [31:0] at_out;
    logic [$clog2(NKV)-1:0] at_head;
    attention #(.LAYERS(NL), .MAX_SEQ(MAX_SEQ), .KV_HEADS(NKV), .HS(HS), .CB(KV4 ? 4 : 8),
                .LANES(HS < 8 ? HS : 8)) attn (
        .clk, .rst, .score_mult(32'(SCORE_MULT)), .score_shift(7'(SCORE_SHIFT)), .kv4(KV4 != 0),
        .kv_we(at_kv_we), .kv_is_v(at_kv_is_v), .kv_layer($clog2(NL)'(layer)), .kv_pos(pos_r),
        .kv_head(at_head), .kv_codes(hcodes), .kv_scale(hscale), .start(at_start), .layer($clog2(NL)'(layer)),
        .head(at_head), .pos(pos_r), .q_codes(hcodes), .q_scale(hscale), .busy(at_busy),
        .out_valid(at_out_valid), .out(at_out), .done(at_done));

    // ---------------- sequencer ----------------
    typedef enum logic [4:0] {
        ST_EMBED, ST_SEEK_L,
        L_RMS_ATT, L_Q_ATT, L_GQ, L_GK, L_GV, L_ROPE_Q, L_ROPE_K, L_KV, L_ATTN, L_Q_O, L_GO, L_ADD_O,
        L_RMS_FFN, L_Q_FFN, L_G1, L_G3, L_SWIGLU, L_Q_H, L_G2, L_ADD_F,
        ST_F_RMS, ST_F_Q, ST_SEEK_C, ST_CLS, ST_DONE
    } step_t;
    step_t step;

    typedef enum logic [5:0] {
        IDLE, DISPATCH, NEXT,
        E_SEEK, E_CODES, E_SSEEK, E_SCALES, E_RD, E_WAIT, E_ISSUE, E_OUT,
        R_RD, R_WAIT, R_G0, R_G1, R_FEED, R_OUT,
        Q_RD, Q_WAIT, Q_FEED, Q_COLLECT, Q_RET,
        G_SCALES, G_CODES, G_START, G_RUN, G_DONE, G_WRITE,
        P_RD0, P_W0, P_RD1, P_W1, P_ISSUE, P_OUT, P_WR1,
        K_START, K_WRITE, A_START, A_LAUNCH, A_COLLECT,
        V_RD0, V_W0, V_RD1, V_W1, V_ISSUE, V_OUT
    } state_t;
    state_t state, q_ret;

    logic [15:0]   tok_r;
    logic [PW-1:0] pos_r;
    logic [4:0]    layer;
    logic [LW-1:0] i, n, oi, q_start, q_len, q_group, ci, gi, qg_cnt;
    logic [2:0]    src, dst, src2;
    logic          q_head_mode;
    // GEMV op
    logic [15:0]   g_rows, g_k, g_tile, g_ntiles, kk;
    logic [LW-1:0] g_kb;
    logic          g_int4, g_cls;
    logic [4:0]    rr;
    logic [5:0]    gg, g_ngroups;
    logic [2:0]    bc;
    logic [31:0]   btmp;
    logic signed [31:0] hold;
    logic [7:0]    gamma_lo;
    logic [$clog2(NH):0] h;
    logic          kv_v;
    logic signed [31:0] best_val;
    logic [15:0]   best_idx;
    logic          best_any;
    logic          bv;               // a GEMV beat is presented

    // ---------------- prefetch FIFO (first-word-fall-through) ----------------
    localparam int FW = $clog2(FIFO_DEPTH);
    logic [7:0]  fmem [FIFO_DEPTH];
    logic [FW-1:0] fw_ptr, fr_ptr;
    logic [FW:0]   f_count;          // bytes in fmem not yet moved to the output register
    logic [7:0]  st_byte;            // FIFO output register
    logic        f_ov;
    wire byte_want = (state == E_CODES) || (state == E_SCALES) || (state == R_G0) || (state == R_G1) ||
                     (state == G_SCALES) || (state == G_CODES);
    wire byte_ok   = f_ov && byte_want;
    wire f_push    = st_valid && st_ready;
    wire f_rd      = (f_count != 0) && (!f_ov || byte_ok);
    always_ff @(posedge clk) begin
        if (f_push) fmem[fw_ptr] <= st_data;
        if (f_rd) st_byte <= fmem[fr_ptr];
        if (rst) begin
            fw_ptr <= '0; fr_ptr <= '0; f_count <= '0; f_ov <= 1'b0;
        end else begin
            if (f_push) fw_ptr <= fw_ptr + 1'b1;
            if (f_rd) fr_ptr <= fr_ptr + 1'b1;
            f_count <= f_count + (FW+1)'(f_push) - (FW+1)'(f_rd);
            f_ov <= f_rd ? 1'b1 : (byte_ok ? 1'b0 : f_ov);
        end
    end

    // segment prefetcher: walks this token's four image segments in consumption order
    logic [1:0]  pf_seg;
    logic        pf_active, pf_seek;
    logic [31:0] pf_left;
    logic [15:0] pf_tok;
    assign st_ready = pf_active && !pf_seek && (pf_left != 0) && (f_count < (FW+1)'(FIFO_DEPTH - 2));
    always_ff @(posedge clk) begin
        seek_req <= 1'b0;
        if (rst) begin
            pf_active <= 1'b0; pf_seek <= 1'b0;
        end else if (state == IDLE && start) begin
            pf_active <= 1'b1; pf_seg <= 2'd0; pf_seek <= 1'b1; pf_tok <= token;
        end else if (pf_active) begin
            if (pf_seek) begin
                seek_req <= 1'b1; pf_seek <= 1'b0;
                case (pf_seg)
                    2'd0: begin seek_addr <= 32'(CLS_BASE) + 32'(pf_tok >> 4) * CLS_TILE_BYTES + CLS_G * 80 + 32'(pf_tok[3:0]) * CLS_KB;
                                pf_left <= 32'(CLS_KB); end
                    2'd1: begin seek_addr <= 32'(EMB_SCALE_BASE) + 32'(pf_tok) * (CLS_G * 5); pf_left <= 32'(CLS_G * 5); end
                    2'd2: begin seek_addr <= 32'(LAYERS_BASE); pf_left <= 32'(IMAGE_BYTES - LAYERS_BASE); end
                    default: begin seek_addr <= 32'(CLS_BASE); pf_left <= 32'(EMB_SCALE_BASE - CLS_BASE); end
                endcase
            end else if (f_push) begin
                pf_left <= pf_left - 1'b1;
                if (pf_left == 32'd1) begin
                    if (pf_seg == 2'd3) pf_active <= 1'b0;
                    else begin pf_seg <= pf_seg + 1'b1; pf_seek <= 1'b1; end
                end
            end
        end
    end
    assign busy = (state != IDLE);

    // GEMV beats, one per cycle: beat kk is presented from memories read with kk's address in the
    // previous cycle; the next address is kk+1 only if beat kk is accepted, so a stall holds the data.
    wire        gv_accept = (state == G_RUN) && bv && gv_in_ready;
    wire [15:0] k_next    = gv_accept ? kk + 1'b1 : kk;
    always_comb begin
        tb_ra = (state == G_RUN) ? (g_int4 ? LW'(k_next >> 1) : LW'(k_next)) : tb_raddr;
        qc_ra = (state == G_RUN) ? LW'(k_next) : qc_raddr;
    end
    assign gv_in_valid   = (state == G_RUN) && bv;
    assign gv_int4       = g_int4;
    assign gv_nib        = kk[0];
    assign gv_chunk_last = ((kk[4:0] == 5'd31) || (kk + 1'b1 == g_k));
    assign gv_tile_last  = (kk + 1'b1 == g_k);
    assign gv_act_scale  = qs[kk >> 6];
    always_comb for (int r = 0; r < 16; r++) begin
        gv_w[r*8 +: 8]       = tb_q[r];
        gv_mult[r*32 +: 32]  = mrf[kk >> 5][r];
        gv_shift[r*8 +: 8]   = srf[kk >> 5][r];
    end

    function automatic logic signed [31:0] nib(input logic [7:0] b, input logic hi);
        return hi ? 32'($signed(b[7:4])) : 32'($signed(b[3:0]));
    endfunction

    task automatic gemv_cfg(input logic [15:0] rows, input logic [15:0] k, input logic int4, input logic [2:0] d, input logic cls);
        g_rows <= rows; g_k <= k; g_int4 <= int4; dst <= d; g_cls <= cls;
        g_ntiles <= (rows + 16'd15) >> 4; g_tile <= '0;
        g_kb <= int4 ? LW'(k >> 1) : LW'(k);
        g_ngroups <= 6'((k + 16'd31) >> 5);
        gg <= '0; rr <= '0; bc <= '0;
        state <= G_SCALES;
    endtask
    task automatic quant_cfg(input logic [2:0] s, input logic [LW-1:0] st, input logic [LW-1:0] len,
                             input logic [LW-1:0] grp, input logic q7, input logic head_mode, input state_t ret);
        src <= s; q_start <= st; q_len <= len; q_group <= grp; aq_qmax7 <= q7; q_head_mode <= head_mode;
        q_ret <= ret; i <= '0; ci <= '0; gi <= '0; qg_cnt <= '0; state <= Q_RD;
    endtask

    always_ff @(posedge clk) begin
        // single-cycle strobes
        wr_en <= 1'b0; gv_start <= 1'b0; aq_in_valid <= 1'b0; rm_in_valid <= 1'b0;
        va_in_valid <= 1'b0; va_op_start <= 1'b0; at_kv_we <= 1'b0; at_start <= 1'b0; done <= 1'b0;
        logit_valid <= 1'b0;
        if (rst) begin
            state <= IDLE;
        end else case (state)
            IDLE: if (start) begin
                tok_r <= token; pos_r <= pos; step <= ST_EMBED; layer <= '0; state <= DISPATCH;
            end
            DISPATCH: case (step)
                ST_EMBED:  state <= E_SEEK;
                ST_SEEK_L: state <= NEXT;      // the prefetcher already moved on to the layers segment
                L_RMS_ATT, L_RMS_FFN, ST_F_RMS: begin src <= BX; dst <= BXB; i <= '0; oi <= '0; state <= R_RD; end
                L_Q_ATT, L_Q_O, L_Q_FFN, ST_F_Q: quant_cfg(BXB, '0, LW'(D), LW'(64), 1'b0, 1'b0, NEXT);
                L_Q_H:     quant_cfg(BH1, '0, LW'(H), LW'(64), 1'b0, 1'b0, NEXT);
                L_GQ:      gemv_cfg(16'(D), 16'(D), WQ4 != 0, BQ, 1'b0);
                L_GK:      gemv_cfg(16'(KVD), 16'(D), WK4 != 0, BK, 1'b0);
                L_GV:      gemv_cfg(16'(KVD), 16'(D), WV4 != 0, BV, 1'b0);
                L_GO:      gemv_cfg(16'(D), 16'(D), WO4 != 0, BT, 1'b0);
                L_G1:      gemv_cfg(16'(H), 16'(D), W14 != 0, BH1, 1'b0);
                L_G3:      gemv_cfg(16'(H), 16'(D), W34 != 0, BH3, 1'b0);
                L_G2:      gemv_cfg(16'(D), 16'(H), W24 != 0, BT, 1'b0);
                ST_CLS:    begin best_any <= 1'b0; gemv_cfg(16'(VOCAB), 16'(D), CLS4 != 0, BT, 1'b1); end
                L_ROPE_Q, L_ROPE_K: begin
                    src <= (step == L_ROPE_Q) ? BQ : BK; n <= LW'((step == L_ROPE_Q) ? D / 2 : KVD / 2);
                    va_op <= 2'd0; va_op_start <= 1'b1; i <= '0; state <= P_RD0;
                end
                L_KV:      begin kv_v <= 1'b0; h <= '0; state <= K_START; end
                L_ATTN:    begin h <= '0; state <= A_START; end
                L_ADD_O, L_ADD_F: begin src <= BX; src2 <= BT; dst <= BX; va_op <= 2'd2; n <= LW'(D); i <= '0; state <= V_RD0; end
                L_SWIGLU:  begin src <= BH1; src2 <= BH3; dst <= BH1; va_op <= 2'd1; n <= LW'(H); i <= '0; state <= V_RD0; end
                ST_SEEK_C: state <= NEXT;
                default: begin token_out <= best_idx; done <= 1'b1; state <= IDLE; end   // ST_DONE
            endcase
            NEXT: begin
                if (step == L_ADD_F && 32'(layer) != NL - 1) begin layer <= layer + 1'b1; step <= L_RMS_ATT; end
                else step <= step_t'(step + 1'b1);
                state <= DISPATCH;
            end

            // ---------------- embedding (tied classifier row + per-group scale) ----------------
            E_SEEK: begin g_kb <= '0; state <= E_CODES; end
            E_CODES: if (byte_ok) begin
                tbuf[0][g_kb] <= st_byte; g_kb <= g_kb + 1'b1;
                if (g_kb == LW'(CLS_KB - 1)) state <= E_SSEEK;
            end
            E_SSEEK: begin gg <= '0; bc <= '0; state <= E_SCALES; end
            E_SCALES: if (byte_ok) begin
                if (bc == 3'd4) begin
                    mrf[gg][0] <= btmp; srf[gg][0] <= st_byte; bc <= '0; gg <= gg + 1'b1;
                    if (gg == 6'(CLS_G - 1)) begin i <= '0; va_op <= 2'd3; state <= E_RD; end
                end else begin btmp <= {st_byte, btmp[31:8]}; bc <= bc + 1'b1; end
            end
            E_RD: begin tb_raddr <= CLS4 ? (i >> 1) : i; state <= E_WAIT; end
            E_WAIT: state <= E_ISSUE;
            E_ISSUE: begin
                va_a <= CLS4 ? nib(tb_q[0], i[0]) : 32'($signed(tb_q[0]));
                va_m <= mrf[i >> 5][0]; va_sh <= 7'(srf[i >> 5][0]); va_in_valid <= 1'b1; state <= E_OUT;
            end
            E_OUT: if (va_out_valid) begin
                wr_en <= 1'b1; wr_addr <= sa(BX, i); wr_data <= va_out0;
                if (i == LW'(D - 1)) state <= NEXT; else begin i <= i + 1'b1; state <= E_RD; end
            end

            // ---------------- RMSNorm (gamma streamed, two bytes per element) ----------------
            R_RD: begin rd_addr <= sa(src, i); state <= R_WAIT; end
            R_WAIT: state <= R_G0;
            R_G0: begin hold <= scr_q; if (byte_ok) begin gamma_lo <= st_byte; state <= R_G1; end end
            R_G1: if (byte_ok) begin
                rm_x <= hold; rm_gamma <= {st_byte, gamma_lo}; rm_in_last <= (i == LW'(D - 1)); rm_in_valid <= 1'b1;
                if (i == LW'(D - 1)) state <= R_OUT; else begin i <= i + 1'b1; state <= R_RD; end
            end
            R_OUT: begin
                if (rm_out_valid) begin wr_en <= 1'b1; wr_addr <= sa(dst, oi); wr_data <= rm_out; oi <= oi + 1'b1; end
                if (rm_done) state <= NEXT;
            end

            // ---------------- act_quant over groups (QC/QS, or one head into hcodes/hscale) ----------------
            Q_RD: begin rd_addr <= sa(src, q_start + i); state <= Q_WAIT; end
            Q_WAIT: state <= Q_FEED;
            Q_FEED: begin
                aq_raw <= scr_q; aq_in_valid <= 1'b1;
                aq_in_last <= (i + 1'b1 == q_len) || (qg_cnt + 1'b1 == q_group);
                i <= i + 1'b1;
                qg_cnt <= (qg_cnt + 1'b1 == q_group) ? '0 : qg_cnt + 1'b1;
                state <= ((i + 1'b1 == q_len) || (qg_cnt + 1'b1 == q_group)) ? Q_COLLECT : Q_RD;
            end
            Q_COLLECT: begin
                if (aq_code_valid) begin
                    if (q_head_mode) hcodes[ci[$clog2(HS)-1:0]*8 +: 8] <= aq_code;
                    else qc[ci] <= aq_code;
                    ci <= ci + 1'b1;
                end
                if (aq_done) begin
                    if (q_head_mode) hscale <= aq_scale; else qs[gi] <= aq_scale;
                    gi <= gi + 1'b1;
                    state <= (i == q_len) ? q_ret : Q_RD;
                end
            end

            // ---------------- GEMV tiles ----------------
            G_SCALES: if (byte_ok) begin
                if (bc == 3'd4) begin
                    mrf[gg][rr] <= btmp; srf[gg][rr] <= st_byte; bc <= '0; rr <= rr + 1'b1;
                    if (rr == 5'd15) begin
                        rr <= '0; gg <= gg + 1'b1;
                        if (gg + 1'b1 == g_ngroups) begin i <= '0; state <= G_CODES; end
                    end
                end else begin btmp <= {st_byte, btmp[31:8]}; bc <= bc + 1'b1; end
            end
            G_CODES: if (byte_ok) begin
                tbuf[rr[3:0]][i] <= st_byte; i <= i + 1'b1;
                if (i + 1'b1 == g_kb) begin
                    i <= '0; rr <= rr + 1'b1;
                    if (rr == 5'd15) begin rr <= '0; state <= G_START; end
                end
            end
            G_START: begin gv_start <= 1'b1; kk <= '0; bv <= 1'b0; state <= G_RUN; end
            G_RUN: begin
                if (!bv) bv <= 1'b1;                       // beat 0 data arrives
                else if (gv_in_ready) begin
                    if (kk + 1'b1 == g_k) begin bv <= 1'b0; state <= G_DONE; end
                    else kk <= kk + 1'b1;
                end
            end
            G_DONE: if (gv_done) begin rr <= '0; state <= G_WRITE; end
            G_WRITE: begin
                logic [15:0] row;
                logic signed [31:0] v;
                row = (g_tile << 4) + 16'(rr);
                v = gv_res[rr[3:0]*32 +: 32];
                if (row < g_rows) begin
                    if (g_cls) begin
                        logit_valid <= 1'b1; logit_idx <= row; logit_val <= v;
                        if (!best_any || v > best_val) begin best_val <= v; best_idx <= row; best_any <= 1'b1; end
                    end else begin
                        wr_en <= 1'b1; wr_addr <= sa(dst, LW'(row)); wr_data <= v;
                    end
                end
                rr <= rr + 1'b1;
                if (rr == 5'd15) begin
                    rr <= '0; gg <= '0; bc <= '0;
                    if (g_tile + 1'b1 == g_ntiles) state <= NEXT;
                    else begin g_tile <= g_tile + 1'b1; state <= G_SCALES; end
                end
            end

            // ---------------- RoPE in place over (x0, x1) pairs ----------------
            P_RD0: begin rd_addr <= sa(src, i << 1); state <= P_W0; end
            P_W0: state <= P_RD1;
            P_RD1: begin hold <= scr_q; rd_addr <= sa(src, (i << 1) + 1'b1); state <= P_W1; end
            P_W1: state <= P_ISSUE;
            P_ISSUE: begin va_a <= hold; va_b <= scr_q; va_in_valid <= 1'b1; state <= P_OUT; end
            P_OUT: if (va_out_valid) begin wr_en <= 1'b1; wr_addr <= sa(src, i << 1); wr_data <= va_out0; state <= P_WR1; end
            P_WR1: begin
                wr_en <= 1'b1; wr_addr <= sa(src, (i << 1) + 1'b1); wr_data <= va_out1;
                if (i + 1'b1 == n) state <= NEXT; else begin i <= i + 1'b1; state <= P_RD0; end
            end

            // ---------------- KV store: quantize each KV head (group HS, qmax 7|127) ----------------
            K_START: begin
                at_head <= h[$clog2(NKV)-1:0];
                quant_cfg(kv_v ? BV : BK, LW'(32'(h) * HS), LW'(HS), LW'(HS), KV4 != 0, 1'b1, K_WRITE);
            end
            K_WRITE: begin
                at_kv_we <= 1'b1; at_kv_is_v <= kv_v;
                if (32'(h) == NKV - 1) begin
                    h <= '0;
                    if (kv_v) state <= NEXT; else begin kv_v <= 1'b1; state <= K_START; end
                end else begin h <= h + 1'b1; state <= K_START; end
            end

            // ---------------- attention heads ----------------
            A_START: quant_cfg(BQ, LW'(32'(h) * HS), LW'(HS), LW'(HS), 1'b0, 1'b1, A_LAUNCH);
            A_LAUNCH: begin at_head <= $clog2(NKV)'(32'(h) / KV_MUL); at_start <= 1'b1; oi <= '0; state <= A_COLLECT; end
            A_COLLECT: begin
                if (at_out_valid) begin wr_en <= 1'b1; wr_addr <= sa(BXB, LW'(32'(h) * HS) + oi); wr_data <= at_out; oi <= oi + 1'b1; end
                if (at_done) begin
                    if (32'(h) == NH - 1) state <= NEXT; else begin h <= h + 1'b1; state <= A_START; end
                end
            end

            // ---------------- two-operand element ops (residual add, SwiGLU) ----------------
            V_RD0: begin rd_addr <= sa(src, i); state <= V_W0; end
            V_W0: state <= V_RD1;
            V_RD1: begin hold <= scr_q; rd_addr <= sa(src2, i); state <= V_W1; end
            V_W1: state <= V_ISSUE;
            V_ISSUE: begin va_a <= hold; va_b <= scr_q; va_in_valid <= 1'b1; state <= V_OUT; end
            V_OUT: if (va_out_valid) begin
                wr_en <= 1'b1; wr_addr <= sa(dst, i); wr_data <= va_out0;
                if (i + 1'b1 == n) state <= NEXT; else begin i <= i + 1'b1; state <= V_RD0; end
            end
            default: state <= IDLE;
        endcase
    end
endmodule
