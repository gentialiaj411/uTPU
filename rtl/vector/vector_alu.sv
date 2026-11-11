// Compatibility shell for legacy focused benches.  Vector arithmetic is
// implemented only by the clocked vector_sequencer; keeping a second
// combinational datapath here would invalidate Phase-3 timing evidence.
module vector_alu (
    input logic signed [383:0] a, b, gamma,
    input logic signed [7:0] ea, eb, eo,
    input logic [5:0] gamma_shift, input logic [5:0] strobe,
    output logic signed [383:0] add_out, norm_out,
    input logic signed [255:0] score, input logic [5:0] valid_count,
    output logic [255:0] softmax_out
);
    assign norm_out = '0;
    assign add_out = '0;
    assign softmax_out = '0;
endmodule
