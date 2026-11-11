// Clocked row arithmetic is owned by vector_sequencer.  This compatibility
// shell keeps the established source list stable while Phase 3 replaces the
// former whole-row combinational implementation with per-operation FSMs.
module vector_serial_alu(
 input logic signed [383:0] a, input logic signed [383:0] gamma,
 input logic signed [255:0] score, input logic [31:0] square_sum, input logic [5:0] word_index,
 input logic [5:0] gamma_shift, input logic [5:0] valid_count,
 output logic [15:0] norm_word, output logic [15:0] soft_word);
 assign norm_word = '0;
 assign soft_word = '0;
endmodule
