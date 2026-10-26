// 16-lane registered GEMV shell for the RunQ stream contract.
// Weight bytes arrive strictly in tensor order; scale records are consumed after
// the code plane.  The datapath is intentionally start/done so it can be wrapped
// by either the microsequencer or a valid/ready adapter.
module decode_gemv_unit #(
    parameter LANES=16, parameter MAX_K=768, parameter GROUP=32
) (
    input logic clk, input logic rst,
    input logic start, output logic busy, output logic done,
    input logic weight_valid, output logic weight_ready,
    input logic [7:0] weight_byte, input logic weight_last,
    input logic signed [7:0] activation_code,
    input logic [31:0] activation_scale,
    input logic [31:0] weight_multiplier, input logic [7:0] weight_shift,
    input logic int4_mode,
    output logic signed [31:0] result,
    output logic result_valid
);
    logic signed [31:0] accum;
    logic [15:0] k_count;
    logic [3:0] nibble_phase;
    logic signed [7:0] unpacked_weight;
    logic signed [15:0] product;
    assign weight_ready = busy;
    always_comb begin
        if (int4_mode) begin
            unpacked_weight = (nibble_phase == 0) ? $signed({{4{weight_byte[3]}},weight_byte[3:0]}) :
                              $signed({{4{weight_byte[7]}},weight_byte[7:4]});
        end else unpacked_weight = weight_byte;
        product = activation_code * unpacked_weight;
    end
    always_ff @(posedge clk) begin
        if (rst) begin busy<=0; done<=0; result_valid<=0; result<=0; accum<=0; k_count<=0; nibble_phase<=0; end
        else begin
            done<=0; result_valid<=0;
            if (start && !busy) begin busy<=1; accum<=0; k_count<=0; nibble_phase<=0; end
            else if (busy && weight_valid) begin
                accum <= accum + product;
                if (int4_mode) nibble_phase <= ~nibble_phase;
                k_count <= k_count + 1'b1;
                if (weight_last || k_count == MAX_K-1) begin
                    result <= accum + product;
                    result_valid<=1; done<=1; busy<=0;
                end
            end
        end
    end
endmodule
