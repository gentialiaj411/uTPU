`timescale 1ns/1ps

// Synth-target compute datapath: pe_array_packed controller + quantizer_array.
// Mirrors the top.sv MAC→requant finalize slice; does NOT include UART/BRAM/FSM.
// Full-chip integration is future work; this module is the realistic packed-DSP synth top.

module top_packed #(
    parameter COMPUTE_DATA_WIDTH     = 8,
    parameter ACCUMULATOR_DATA_WIDTH = 32,
    parameter ARRAY_SIZE             = 8,
    parameter ARRAY_SIZE_WIDTH       = $clog2(ARRAY_SIZE),
    parameter BUFFER_WORD_SIZE       = 16,
    parameter NUM_COMPUTE_LANES      = ARRAY_SIZE*ARRAY_SIZE,
    parameter MAX_BATCH_COUNT        = 64,
    parameter MAX_BATCH_COUNT_WIDTH  = $clog2(MAX_BATCH_COUNT + 1),
    // Match top.sv Step-2 default: one column of lanes. Override to N^2 for A/B.
    parameter QUANTIZER_LANES        = ARRAY_SIZE,
    parameter QUANTIZER_SIZE         = QUANTIZER_LANES,
    parameter QUANTIZER_SIZE_WIDTH   = $clog2(QUANTIZER_SIZE)
) (
    input  logic clk,
    input  logic rst,
    input  logic compute,
    input  logic load_en,
    input  logic [MAX_BATCH_COUNT_WIDTH-1:0] batch_count,
    output logic compute_done,
    input  logic signed [COMPUTE_DATA_WIDTH-1:0] weights_in [NUM_COMPUTE_LANES-1:0],
    input  logic signed [COMPUTE_DATA_WIDTH-1:0] datas_arr   [ARRAY_SIZE*MAX_BATCH_COUNT-1:0],
    input  logic requant_enable,
    input  logic [(QUANTIZER_SIZE*16)-1:0] requant_multiplier_flat,
    input  logic [(QUANTIZER_SIZE*16)-1:0] requant_right_shift_flat,
    output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] accum_results [NUM_COMPUTE_LANES-1:0],
    output logic signed [COMPUTE_DATA_WIDTH-1:0]     quant_results [QUANTIZER_SIZE-1:0]
);

    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] stream_out [ARRAY_SIZE*MAX_BATCH_COUNT-1:0];
    logic [(QUANTIZER_SIZE*ACCUMULATOR_DATA_WIDTH)-1:0] quantizer_ins_flat;
    logic [(QUANTIZER_SIZE*COMPUTE_DATA_WIDTH)-1:0] quantizer_out_flat;

    pe_controller_packed #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .ARRAY_SIZE_WIDTH(ARRAY_SIZE_WIDTH),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
        .NUM_COMPUTE_LANES(NUM_COMPUTE_LANES),
        .MAX_BATCH_COUNT(MAX_BATCH_COUNT),
        .BATCH_COUNT_WIDTH(MAX_BATCH_COUNT_WIDTH)
    ) u_compute (
        .clk(clk),
        .rst(rst),
        .compute(compute),
        .load_en(load_en),
        .batch_count(batch_count),
        .done(compute_done),
        .datas_arr(datas_arr),
        .weights_in(weights_in),
        .results_arr(stream_out)
    );

    quantizer_array #(
        .QUANTIZER_SIZE(QUANTIZER_SIZE),
        .QUANTIZER_SIZE_WIDTH(QUANTIZER_SIZE_WIDTH),
        .ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
        .COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH)
    ) u_quantizer_array (
        .ins_flat(quantizer_ins_flat),
        .requant_enable(requant_enable),
        .requant_multiplier_flat(requant_multiplier_flat),
        .requant_right_shift_flat(requant_right_shift_flat),
        .results_flat(quantizer_out_flat)
    );

    genvar qi;
    generate
        for (qi = 0; qi < QUANTIZER_SIZE; qi++) begin : pack_lanes
            localparam int QI_IN_LO = qi * ACCUMULATOR_DATA_WIDTH;
            localparam int QI_OUT_LO = qi * COMPUTE_DATA_WIDTH;
            assign quantizer_ins_flat[QI_IN_LO +: ACCUMULATOR_DATA_WIDTH] = stream_out[qi];
            assign accum_results[qi] = stream_out[qi];
            assign quant_results[qi] = quantizer_out_flat[QI_OUT_LO +: COMPUTE_DATA_WIDTH];
        end
    endgenerate

endmodule
