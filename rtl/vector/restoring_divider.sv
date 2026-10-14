// Exact unsigned restoring divider for the frozen VSOFTMAX quotient contract.
// One quotient bit is retired per clock after a start pulse.
module restoring_divider #(
    parameter NUM_WIDTH = 24,
    parameter DEN_WIDTH = 22
) (
    input logic clk, input logic rst, input logic start,
    input logic [NUM_WIDTH-1:0] numerator,
    input logic [DEN_WIDTH-1:0] denominator,
    output logic busy, done,
    output logic [NUM_WIDTH-1:0] quotient
);
    logic [NUM_WIDTH-1:0] dividend;
    logic [DEN_WIDTH-1:0] divisor;
    logic [DEN_WIDTH:0] remainder;
    logic [$clog2(NUM_WIDTH)-1:0] bit_index;
    logic [DEN_WIDTH:0] trial_remainder;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 0; done <= 0; quotient <= 0; dividend <= 0;
            divisor <= 0; remainder <= 0; bit_index <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                quotient <= 0;
                dividend <= numerator;
                divisor <= denominator;
                remainder <= 0;
                bit_index <= NUM_WIDTH-1;
                busy <= (denominator != 0);
                done <= (denominator == 0);
            end else if (busy) begin
                trial_remainder = {remainder[DEN_WIDTH-1:0], dividend[bit_index]};
                if (trial_remainder >= {1'b0, divisor}) begin
                    remainder <= trial_remainder - {1'b0, divisor};
                    quotient[bit_index] <= 1'b1;
                end else begin
                    remainder <= trial_remainder;
                    quotient[bit_index] <= 1'b0;
                end
                if (bit_index == 0) begin
                    busy <= 0;
                    done <= 1;
                end else bit_index <= bit_index - 1'b1;
            end
        end
    end
endmodule
