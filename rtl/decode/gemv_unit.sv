// 16-row decode GEMV tile, bit-exact with numerics_v2 runq_runtime._linear_high.
//
// One k step per accepted beat: an activation code and one weight byte per row
// (INT8 code, or two INT4 codes with nibble_hi selecting k odd). Raw INT32
// dot products accumulate per chunk; a chunk ends at every weight-group or
// activation-group-64 boundary (chunk_last). At chunk end the 16 partial sums
// are snapshotted and one shared scaler pipeline processes them row by row:
//   small = rshift_rne(multiplier, 16)
//   p     = acc * act_scale * small
//   s     = rshift_rne(p, shift - 16)
//   out  += div127(s)          (Q32 reciprocal estimate + one +/-1 correction)
// Chunks shorter than LANES beats back-pressure via in_ready.
module decode_gemv_unit #(
    parameter int LANES = 16
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic                    int4_mode,
    input  logic                    nibble_hi,
    input  logic signed [7:0]       act_code,
    input  logic [LANES*8-1:0]      weight_bytes,
    input  logic                    chunk_last,
    input  logic                    tile_last,
    input  logic signed [31:0]      act_scale,
    input  logic [LANES*32-1:0]     multipliers,
    input  logic [LANES*8-1:0]      shifts,
    output logic                    busy,
    output logic                    done,
    output logic [LANES*32-1:0]     result
);
    localparam int RW = $clog2(LANES);
    localparam logic [25:0] INV127_Q32 = 26'd33818641;
    localparam int PIPE = 6;

    function automatic logic signed [63:0] rshift_rne(input logic signed [63:0] x, input logic [5:0] s);
        logic [63:0] mag, q, rem, half;
        logic inc;
        if (s == 0) return x;
        mag  = x[63] ? 64'(-x) : 64'(x);
        q    = mag >> s;
        rem  = mag & ((64'd1 << s) - 64'd1);
        half = 64'd1 << (s - 6'd1);
        inc  = (rem > half) || ((rem == half) && q[0]);
        q    = q + 64'(inc);
        return x[63] ? -$signed(q) : $signed(q);
    endfunction

    // ---------------- MAC lanes ----------------
    logic signed [31:0] acc [LANES];
    logic signed [15:0] prod [LANES];
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            logic [7:0] b;
            logic signed [7:0] w;
            b = weight_bytes[i*8 +: 8];
            if (int4_mode) w = nibble_hi ? {{4{b[7]}}, b[7:4]} : {{4{b[3]}}, b[3:0]};
            else           w = b;
            prod[i] = act_code * w;
        end
    end

    // ---------------- chunk snapshot ----------------
    logic                snap_busy, snap_tile_last;
    logic [RW-1:0]       snap_row;
    logic signed [31:0]  snap_acc [LANES];
    logic [31:0]         snap_mult [LANES];
    logic [7:0]          snap_shift [LANES];
    logic signed [31:0]  snap_act;
    logic                draining;   // final chunk captured, waiting for scaler

    assign in_ready = busy && !draining && !(chunk_last && snap_busy);
    wire   beat     = in_valid && in_ready;

    // ---------------- scaler pipeline ----------------
    logic [PIPE-1:0]     v;
    logic [RW-1:0]       r1, r2, r3, r4, r5;
    logic signed [63:0]  m1;            // acc * act_scale
    logic signed [63:0]  small1;
    logic [5:0]          sh1, sh2;
    logic signed [63:0]  p2;            // m1 * small
    logic signed [63:0]  s3;            // rshift_rne(p, shift-16)
    logic [63:0]         mag4;
    logic                neg4, neg5;
    logic [63:0]         q4;            // reciprocal estimate
    logic [63:0]         q5;            // corrected magnitude
    logic signed [63:0]  out_acc [LANES];
    logic                last_issue;    // final chunk's last row entered the pipe
    logic [PIPE-1:0]     fin;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 0; done <= 0; draining <= 0;
            snap_busy <= 0; snap_tile_last <= 0; snap_row <= '0;
            v <= '0; fin <= '0;
            for (int i = 0; i < LANES; i++) begin acc[i] <= 0; out_acc[i] <= 0; end
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1; draining <= 1'b0;
                for (int i = 0; i < LANES; i++) begin acc[i] <= 0; out_acc[i] <= 0; end
            end

            // MAC + chunk capture
            if (beat) begin
                for (int i = 0; i < LANES; i++) acc[i] <= chunk_last ? 32'sd0 : acc[i] + prod[i];
                if (chunk_last) begin
                    for (int i = 0; i < LANES; i++) begin
                        snap_acc[i]   <= acc[i] + prod[i];
                        snap_mult[i]  <= multipliers[i*32 +: 32];
                        snap_shift[i] <= shifts[i*8 +: 8];
                    end
                    snap_act <= act_scale;
                    snap_busy <= 1'b1; snap_row <= '0; snap_tile_last <= tile_last;
                    if (tile_last) draining <= 1'b1;
                end
            end

            // Stage 1: issue one row from the snapshot.
            v[0] <= snap_busy; fin[0] <= snap_busy && snap_tile_last && (snap_row == RW'(LANES-1));
            if (snap_busy) begin
                r1     <= snap_row;
                m1     <= 64'(snap_acc[snap_row]) * 64'(snap_act);
                small1 <= rshift_rne(64'($signed(snap_mult[snap_row])), 6'd16);
                sh1    <= 6'(snap_shift[snap_row] - 8'd16);
                if (snap_row == RW'(LANES-1)) snap_busy <= 1'b0;
                snap_row <= snap_row + 1'b1;
            end
            // Stage 2: second scale multiply.
            v[1] <= v[0]; fin[1] <= fin[0]; r2 <= r1; sh2 <= sh1;
            p2 <= m1 * small1;
            // Stage 3: round-to-nearest-even shift.
            v[2] <= v[1]; fin[2] <= fin[1]; r3 <= r2;
            s3 <= rshift_rne(p2, sh2);
            // Stage 4: reciprocal estimate of |s|/127, rounded half up.
            v[3] <= v[2]; fin[3] <= fin[2]; r4 <= r3;
            neg4 <= s3[63];
            mag4 <= s3[63] ? 64'(-s3) : 64'(s3);
            q4   <= 64'(((96'(s3[63] ? 64'(-s3) : 64'(s3)) * 96'(INV127_Q32)) + (96'd1 << 31)) >> 32);
            // Stage 5: one-step correction to the nearest quotient (127 is odd: no ties).
            v[4] <= v[3]; fin[4] <= fin[3]; r5 <= r4; neg5 <= neg4;
            begin
                logic signed [65:0] err2;
                err2 = 66'sd2 * ($signed({2'b00, mag4}) - $signed({2'b00, q4}) * 66'sd127);
                if (err2 > 66'sd127)       q5 <= q4 + 64'd1;
                else if (err2 < -66'sd127) q5 <= q4 - 64'd1;
                else                       q5 <= q4;
            end
            // Stage 6: accumulate.
            v[5] <= v[4]; fin[5] <= fin[4];
            if (v[4]) out_acc[r5] <= out_acc[r5] + (neg5 ? -$signed(q5) : $signed(q5));
            if (v[5] && fin[5]) begin busy <= 1'b0; draining <= 1'b0; done <= 1'b1; end
        end
    end

    logic [LANES*32-1:0] result_int;
    always_comb for (int i = 0; i < LANES; i++) result_int[i*32 +: 32] = out_acc[i][31:0];
`ifdef GEMV_MUTATE
    assign result = result_int ^ ({{(LANES*32-1){1'b0}}, 1'b1} << (3*32));
`else
    assign result = result_int;
`endif
endmodule
