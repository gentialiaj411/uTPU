// Legacy direct-arithmetic test shim.  The synthesized implementation is the
// registered one-word vector_sequencer pipeline; this module retains only the
// statically unrolled VADD oracle interface used by focused RTL tests.
module vector_alu (
    input logic signed [383:0] a, b, gamma,
    input logic signed [7:0] ea, eb, eo,
    input logic [5:0] gamma_shift, input logic [5:0] strobe,
    output logic signed [383:0] add_out, norm_out,
    input logic signed [255:0] score, input logic [5:0] valid_count,
    output logic [255:0] softmax_out
);
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
    genvar lane;
    generate
        for (lane = 0; lane < 48; lane = lane + 1) begin: g_vadd_oracle
            always @* add_out[lane*8 +: 8] = sat8(
                rne_shift($signed(a[lane*8 +: 8]), $signed(eo)-$signed(ea)) +
                rne_shift($signed(b[lane*8 +: 8]), $signed(eo)-$signed(eb)));
        end
    endgenerate
    // Norm and softmax moved to the clocked serial unit.  Their old direct
    // combinational oracle deliberately has no implementation here.
    assign norm_out = '0;
    assign softmax_out = '0;
endmodule
