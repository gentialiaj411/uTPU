// Combinational reference datapath for differential simulation.  Ports are
// packed deliberately: Icarus does not reliably schedule changes propagated
// through unpacked array ports, while the synthesizable sequencer keeps the
// area-oriented one-word interface.
module vector_alu (
    input logic signed [383:0] a, b, gamma,
    input logic signed [7:0] ea, eb, eo,
    input logic [5:0] gamma_shift, input logic [5:0] strobe,
    output logic signed [383:0] add_out, norm_out,
    input logic signed [255:0] score, input logic [5:0] valid_count,
    output logic [255:0] softmax_out
);
    logic [15:0] exp_q15 [0:128];
    logic [15:0] rsqrt_q14 [0:191];
    integer i;
    initial begin
`ifdef SYNTHESIS
        $readmemh("exp_q15.mem", exp_q15);
        $readmemh("rsqrt_q14.mem", rsqrt_q14);
`else
        $readmemh("rtl/vector/exp_q15.mem", exp_q15);
        $readmemh("rtl/vector/rsqrt_q14.mem", rsqrt_q14);
`endif
    end
    function automatic signed [63:0] rne_shift(input signed [63:0] value, input integer shift);
        reg signed [63:0] mag, q, rem, half;
        begin
            if (shift <= 0) rne_shift = value;
            else begin
                mag = value < 0 ? -value : value; q = mag >>> shift;
                rem = mag - (q <<< shift); half = 64'sd1 <<< (shift - 1);
                if ((rem > half) || ((rem == half) && q[0])) q = q + 1;
                rne_shift = value < 0 ? -q : q;
            end
        end
    endfunction
    function automatic signed [7:0] sat8(input signed [63:0] value);
        begin
            if (value > 127) sat8 = 8'sd127;
            else if (value < -128) sat8 = -8'sd128;
            else sat8 = value[7:0];
        end
    endfunction
    integer sumsq, mean_q, mean_rem, exponent, mantissa, bin, force_strobe;
    integer max_score, rel, total, numerator;
    reg signed [63:0] r0, t, r1, product;
    always_comb begin
        force_strobe = strobe;
        for (i = 0; i < 48; i = i + 1)
            add_out[i*8 +: 8] = sat8(rne_shift($signed(a[i*8 +: 8]), $signed(eo)-$signed(ea)) +
                                      rne_shift($signed(b[i*8 +: 8]), $signed(eo)-$signed(eb)));
        sumsq = 0;
        for (i = 0; i < 48; i = i + 1)
            sumsq = sumsq + $signed(a[i*8 +: 8]) * $signed(a[i*8 +: 8]);
        mean_q = sumsq / 48; mean_rem = sumsq % 48;
        if ((2 * mean_rem > 48) || ((2 * mean_rem == 48) && (mean_q & 1))) mean_q = mean_q + 1;
        mean_q = mean_q + 1; exponent = 0;
        for (i = 0; i < 31; i = i + 1)
            if ((mean_q >= (1 << i)) && ((i & 1) == 0)) exponent = i;
        mantissa = rne_shift($signed(mean_q) <<< 14, exponent);
        if (mantissa < 16384) mantissa = 16384;
        if (mantissa > 65535) mantissa = 65535;
        bin = (mantissa - 16384) >>> 8; if (bin > 191) bin = 191;
        r0 = rsqrt_q14[bin]; t = rne_shift($signed(mantissa) * r0 * r0, 28);
        r1 = rne_shift(r0 * (3*16384 - t), 15);
        for (i = 0; i < 48; i = i + 1) begin
            product = $signed(a[i*8 +: 8]) * $signed(gamma[i*8 +: 8]) * r1;
            norm_out[i*8 +: 8] = sat8(rne_shift(product, 14 + gamma_shift + (exponent >>> 1)));
        end
        max_score = -128;
        for (i = 0; i < 32; i = i + 1)
            if (i < valid_count && $signed(score[i*8 +: 8]) > max_score) max_score = $signed(score[i*8 +: 8]);
        total = 0;
        for (i = 0; i < 32; i = i + 1) if (i < valid_count) begin
            rel = $signed(score[i*8 +: 8]) - max_score;
            if (rel < -128) rel = -128; if (rel > 0) rel = 0;
            total = total + exp_q15[-rel];
        end
        for (i = 0; i < 32; i = i + 1) begin
            softmax_out[i*8 +: 8] = 0;
            if (i < valid_count) begin
                rel = $signed(score[i*8 +: 8]) - max_score;
                if (rel < -128) rel = -128; if (rel > 0) rel = 0;
                numerator = 255 * exp_q15[-rel] + total / 2;
                softmax_out[i*8 +: 8] = numerator / total;
            end
        end
    end
endmodule
