`timescale 1ns/1ps

// Iverilog-compatible LeakyReLU array with packed ports
module leaky_relu_array #(
    parameter RELU_SIZE          = 64,
    parameter RELU_SIZE_WIDTH    = $clog2(RELU_SIZE),
    parameter ALPHA              = 2,
    parameter COMPUTE_DATA_WIDTH = 4
) ( 
    // IVERILOG FIX: Use packed vectors instead of unpacked arrays
    input  logic [RELU_SIZE*COMPUTE_DATA_WIDTH-1:0] in_packed,
    output logic [RELU_SIZE*COMPUTE_DATA_WIDTH-1:0] result_packed
);

    // Internal unpacked arrays
    logic signed [COMPUTE_DATA_WIDTH-1:0] in     [RELU_SIZE-1:0];
    logic signed [COMPUTE_DATA_WIDTH-1:0] result [RELU_SIZE-1:0];

    // Unpack inputs
    genvar gi;
    generate
        for (gi = 0; gi < RELU_SIZE; gi++) begin: gen_unpack
            assign in[gi] = in_packed[gi*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH];
            assign result_packed[gi*COMPUTE_DATA_WIDTH +: COMPUTE_DATA_WIDTH] = result[gi];
        end
    endgenerate

    // Individual LeakyReLU instances
    genvar i;
    generate
        for (i = 0; i < RELU_SIZE; i++) begin: array_gen
            leaky_relu #(
                .ALPHA(ALPHA),
                .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
            ) u_relu (
                .in(in[i]),
                .result(result[i])
            );
        end
    endgenerate
    
endmodule: leaky_relu_array
