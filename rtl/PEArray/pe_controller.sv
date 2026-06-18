`timescale 1ns/1ps

module pe_controller #(
	parameter ARRAY_SIZE 		 = 8,
	parameter ARRAY_SIZE_WIDTH 	 = $clog2(ARRAY_SIZE),
	parameter COMPUTE_DATA_WIDTH     = 4,
	parameter ACCUMULATOR_DATA_WIDTH = 16,
	parameter BUFFER_WORD_SIZE       = 16,
	parameter NUM_COMPUTE_LANES      = BUFFER_WORD_SIZE/COMPUTE_DATA_WIDTH,
	parameter MAX_BATCH_COUNT        = 64,
	parameter BATCH_COUNT_WIDTH      = $clog2(MAX_BATCH_COUNT + 1)
    ) (
	input  logic clk, rst, compute, load_en,
	input  logic [BATCH_COUNT_WIDTH-1:0] batch_count,
	output logic done,
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_arr   [ARRAY_SIZE*MAX_BATCH_COUNT-1:0],
	input  logic signed [COMPUTE_DATA_WIDTH-1:0]     weights_in  [ARRAY_SIZE*ARRAY_SIZE-1:0],
	output logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results_arr [ARRAY_SIZE*MAX_BATCH_COUNT-1:0]
    );

    localparam MAX_CYCLE_LENGTH = (ARRAY_SIZE * 2) + MAX_BATCH_COUNT - 1;

    logic        [$clog2(MAX_CYCLE_LENGTH+1)-1:0] cycle_count;
    logic signed [COMPUTE_DATA_WIDTH-1:0]     datas_in [ARRAY_SIZE-1:0];
    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] results  [ARRAY_SIZE-1:0];
    logic                                      compute_complete_hold;

    int i;

    always_ff @(posedge clk) begin
	done <= '0;
	if (rst) begin
	    cycle_count <= '0;
            compute_complete_hold <= 1'b0;
	    for (i=0; i < ARRAY_SIZE; i++)
		datas_in[i] <= '0;
            for (i=0; i < ARRAY_SIZE*MAX_BATCH_COUNT; i++)
                results_arr[i] <= '0;
	end else begin
            if (compute) begin
                int active_columns;
                int capture_cycle;
                int done_cycle;
                active_columns = (batch_count > 0) ? batch_count : 1;
                if (compute_complete_hold) begin
                    cycle_count <= '0;
                    for (i = 0; i < ARRAY_SIZE; i++)
                        datas_in[i] <= '0;
                end else begin
                    capture_cycle = cycle_count + 1;
                    done_cycle = (ARRAY_SIZE * 2) + active_columns - 1;
                    if (capture_cycle >= done_cycle) begin
                        cycle_count <= '0;
                        compute_complete_hold <= 1'b1;
                    end else begin
                        cycle_count <= cycle_count + 1'b1;
                    end

                    for (i = 0; i < ARRAY_SIZE; i++) begin
                        if ((cycle_count < active_columns + i) && (cycle_count >= i))
                            datas_in[i] <= datas_arr[ARRAY_SIZE*(cycle_count - i) + i];
                        else
                            datas_in[i] <= '0;
                    end

                    for (int col = 0; col < MAX_BATCH_COUNT; col++) begin
                        if (col < active_columns) begin
                            for (int row = 0; row < ARRAY_SIZE; row++) begin
                                if ((ARRAY_SIZE + 1 + row + col) == capture_cycle) begin
`ifdef ICARUS
                                    logic signed [ACCUMULATOR_DATA_WIDTH-1:0] captured_result;
                                    captured_result = $signed(u_pe_array.accumulators[ARRAY_SIZE-1][row]);
                                    if (^u_pe_array.accumulators[ARRAY_SIZE-1][row] === 1'bx)
                                        captured_result = '0;
                                    results_arr[(col * ARRAY_SIZE) + row] <= captured_result;
`else
                                    results_arr[(col * ARRAY_SIZE) + row] <= results[row];
`endif
                                    if ((row == ARRAY_SIZE-1) && (col == active_columns-1))
			                done <= 1'b1;
                                end
                            end
                        end
                    end
                end
            end else begin
                cycle_count <= '0;
                compute_complete_hold <= 1'b0;
                for (i=0; i < ARRAY_SIZE; i++)
                    datas_in[i] <= '0;
            end
	end
    end


    pe_array #(
	.ARRAY_SIZE(ARRAY_SIZE),
	.ARRAY_SIZE_WIDTH(ARRAY_SIZE_WIDTH),
	.COMPUTE_DATA_WIDTH(COMPUTE_DATA_WIDTH),
	.ACCUMULATOR_DATA_WIDTH(ACCUMULATOR_DATA_WIDTH),
	.BUFFER_WORD_SIZE(BUFFER_WORD_SIZE),
	.NUM_COMPUTE_LANES(NUM_COMPUTE_LANES)
    ) u_pe_array (
	.clk(clk),
	.rst(rst),
	.compute(compute),
	.load_en(load_en),
	.datas_in(datas_in),
	.weights_in(weights_in),
	.results(results)
    );

endmodule: pe_controller
