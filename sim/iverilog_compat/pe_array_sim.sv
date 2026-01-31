/*
*  PE/MXU ARRAY Module - Iverilog Compatible Version
*  Fixed the out-of-bounds array access for i-1 when i=0
*/

`timescale 1ns/1ps

module pe_array #(
    parameter ARRAY_SIZE             = 8,
    parameter ARRAY_SIZE_WIDTH       = $clog2(ARRAY_SIZE),
    parameter COMPUTE_DATA_WIDTH     = 4,
    parameter ACCUMULATOR_DATA_WIDTH = 16,
    parameter BUFFER_WORD_SIZE       = 16,
    parameter NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH
) (
    input  logic clk, rst, compute, load_en,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in   [ARRAY_SIZE-1:0],
    input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in [ARRAY_SIZE*ARRAY_SIZE-1:0],
    output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results    [ARRAY_SIZE-1:0]
);
    
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] accumulators [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0]     activations  [ARRAY_SIZE-1:0][ARRAY_SIZE:0];
    
    // Helper signal for partial sum inputs (avoids out-of-bounds access)
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] partial_sum_wire [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];

    genvar i, j;
    generate 
        for (i = 0; i < ARRAY_SIZE; i++) begin: connect_ins
            assign activations[i][0] = datas_in[i];
        end

        for (i = 0; i < ARRAY_SIZE; i++) begin: connect_results
            assign results[i] = accumulators[ARRAY_SIZE-1][i];
        end

        // Generate partial sum wires to avoid out-of-bounds access
        for (i = 0; i < ARRAY_SIZE; i++) begin: gen_partial_sums
            for (j = 0; j < ARRAY_SIZE; j++) begin: gen_partial_sum_cols
                if (i == 0) begin: first_row
                    assign partial_sum_wire[i][j] = '0;
                end else begin: other_rows
                    assign partial_sum_wire[i][j] = accumulators[i-1][j];
                end
            end
        end

        for (i = 0; i < ARRAY_SIZE; i++) begin: gen_rows
            for (j = 0; j < ARRAY_SIZE; j++) begin: gen_cols
                pe #(
                    COMPUTE_DATA_WIDTH,
                    ACCUMULATOR_DATA_WIDTH
                ) u_pe (
                    .clk(clk),
                    .rst(rst),
                    .compute(compute),
                    .load_en(load_en),
                    .data_in(activations[i][j]),
                    .weight_in(weights_in[i*ARRAY_SIZE+j]),
                    .partial_sum_in(partial_sum_wire[i][j]),
                    .data_out(activations[i][j+1]),
                    .partial_sum_out(accumulators[i][j])
                );
            end
        end
    endgenerate
endmodule: pe_array
