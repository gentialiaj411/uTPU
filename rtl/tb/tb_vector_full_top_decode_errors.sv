`timescale 1ns/1ps
// Full-chip vector decoder negative tests.  The program words are placed in
// the instruction BRAM, then execution enters through the normal fetch,
// escape and descriptor states; no vector-unit port is driven by this TB.
module tb_vector_full_top_decode_errors;
    parameter integer TB_VECTOR_EN = 1;
    parameter integer TB_CASE = 0;
    logic clk = 0, rst = 0, rx = 1;
    wire tx, led_rst;
    localparam integer FIFO_WIDTH = 256;
    localparam integer PTR_W = $clog2(FIFO_WIDTH);
    integer tests = 0, errors = 0, total_cycles = 0;
    logic [15:0] prog_image [0:255];

    top #(.ARRAY_SIZE(8), .BUFFER_SIZE(4096), .COMPUTE_DATA_WIDTH(8),
          .ACCUMULATOR_DATA_WIDTH(32), .EXT_ADDR_EN(1), .PROG_DEPTH(256),
          .FIFO_WIDTH(FIFO_WIDTH), .MAX_BATCH_COUNT(4),
          .QUANTIZER_PIPE_DEPTH(1), .VECTOR_UNIT_EN(TB_VECTOR_EN)) dut
        (.clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst));
    always #5 clk = ~clk;

    task automatic check(input string name, input bit condition);
        begin
            tests = tests + 1;
            if (!condition) begin errors = errors + 1; $display("[FAIL] %s", name); end
            else $display("[ OK ] %s", name);
        end
    endtask
    task automatic wait_cycles(input integer count);
        repeat (count) @(posedge clk);
    endtask
    task automatic push_rx(input logic [7:0] byte_value);
        @(posedge clk);
        dut.fifo_in.mem[dut.fifo_in.w_ptr[PTR_W-1:0]] = byte_value;
        dut.fifo_in.w_ptr = dut.fifo_in.w_ptr + 1'b1;
        @(posedge clk);
    endtask
    task automatic upload_program(input integer words);
        integer wi;
        begin
            push_rx(8'ha1); push_rx(words[7:0]); push_rx(words[15:8]);
            for (wi = 0; wi < words; wi = wi + 1) begin
                push_rx(prog_image[wi][7:0]); push_rx(prog_image[wi][15:8]);
            end
        end
    endtask
    task automatic wait_state(input logic [5:0] state_value, input integer limit, output bit seen);
        integer n;
        begin
            seen = 0;
            for (n = 0; n < limit && !seen; n = n + 1) begin
                @(posedge clk); total_cycles = total_cycles + 1;
                if (dut.current_state == state_value) seen = 1;
            end
        end
    endtask
    function automatic [15:0] peek(input integer address);
        peek = dut.u_unified_buffer.bank_mem[address % 32][address / 32];
    endfunction
    task automatic poke(input integer address, input logic [15:0] value);
        dut.u_unified_buffer.bank_mem[address % 32][address / 32] = value;
    endtask

    // id selects a distinct Section-5 error class.  The descriptor is valid
    // enough to reach the production vector sequencer whenever the error is
    // a record/pointer/alias error, which must report code 04 rather than the
    // header decoder's code 03.
    task automatic setup_case(input integer id, output [7:0] expect_code, output integer words);
        integer i;
        logic [15:0] h;
        begin
            for (i = 0; i < 256; i = i + 1) prog_image[i] = 16'h0004;
            for (i = 0; i < 256; i = i + 1) poke(i, 16'h5a5a);
            h = 16'h0809; // VADD, parameter kind 1
            words = 7;
            expect_code = 8'h04;
            prog_image[0] = 16'h05a5;
            prog_image[1] = h;
            prog_image[2] = 16'd100;
            prog_image[3] = 16'd200;
            prog_image[4] = 16'd300;
            prog_image[5] = 16'd48;
            prog_image[6] = 16'h0004;
            // Valid VADD record and source data; negative cases below alter it.
            poke(300, 16'h0001); poke(301, 16'd400); poke(302, 16'h0000); poke(303, 16'h0000);
            case (id)
                0: begin prog_image[5] = 0; expect_code = 8'h03; end // zero length
                1: begin words = 3; expect_code = 8'h03; end // truncated descriptor
                2: begin prog_image[1] = 16'h3869; expect_code = 8'h03; end // reserved subop
                3: begin prog_image[1] = 16'h1009; expect_code = 8'h03; end // param kind
                4: begin poke(300, 16'h0011); end // reserved record bit
                5: begin poke(302, 16'h0020); end // exponent outside [-31,31]
                6: begin poke(301, 16'd4090); end // src1 pointer extent OOR
                7: begin prog_image[5] = 16'd50; expect_code = 8'h03; end // v1 shape
                8: begin prog_image[1] = 16'h0909; expect_code = 8'h03; end // illegal flag
                9: begin poke(300, 16'h0002); end // bad record kind
                10: begin prog_image[2] = 16'd101; prog_image[3] = 16'd100; end // partial dst/src overlap
                11: begin
                    prog_image[1] = 16'h2139; // softmax, kind 4, causal
                    prog_image[2] = 16'd100; prog_image[3] = 16'd200;
                    prog_image[4] = 16'd300; prog_image[5] = 16'd1024;
                    poke(300, 16'h0004); poke(301, 16'd700);
                    prog_image[1] = 16'h2739; // head_id=3
                    expect_code = 8'h03;
                end
                12: begin
                    prog_image[1] = 16'h2039; // softmax, noncausal
                    prog_image[2] = 16'd100; prog_image[3] = 16'd200;
                    prog_image[4] = 16'd300; prog_image[5] = 16'd1024;
                    poke(300, 16'h0004); poke(301, 16'd700); expect_code = 8'h03;
                end
            endcase
        end
    endtask

    task automatic run_enabled_case(input integer id, input string name);
        bit halt_seen;
        integer i;
        logic [7:0] code;
        integer words;
        begin
            setup_case(id, code, words);
            upload_program(words);
            wait_state(dut.WAIT_START_STATE, 10000, halt_seen);
            check({name, " upload reaches WAIT_START"}, halt_seen);
            push_rx(8'ha2);
            wait_state(dut.HALT_STATE, 10000, halt_seen);
            check({name, " reached HALT"}, halt_seen);
            check({name, " sticky perf/status code"}, dut.perf_busy_status_word[63:56] === code);
            for (i = 0; i < 48; i = i + 1)
                check({name, " no output write"}, peek(100+i) === 16'h5a5a);
        end
    endtask

    task automatic run_absent_case;
        bit halt_seen;
        integer i;
        begin
            for (i = 0; i < 256; i = i + 1) prog_image[i] = 16'h0004;
            for (i = 0; i < 32; i = i + 1) poke(i, 16'ha55a);
            // If decoded as legacy after the escape, these are BSTORE header,
            // address, count, payload and would overwrite buffer[0].
            prog_image[0] = 16'h05a5;
            prog_image[1] = 16'h0006;
            prog_image[2] = 16'h0000;
            prog_image[3] = 16'h0001;
            prog_image[4] = 16'hbeef;
            prog_image[5] = 16'h0004;
            upload_program(6); wait_state(dut.WAIT_START_STATE, 10000, halt_seen);
            check("vector absent upload reaches WAIT_START", halt_seen);
            push_rx(8'ha2); wait_state(dut.HALT_STATE, 10000, halt_seen);
            check("vector absent escape reaches HALT", halt_seen);
            check("vector absent sticky perf/status code", dut.perf_busy_status_word[63:56] === 8'h01);
            for (i = 0; i < 32; i = i + 1)
                check("vector absent descriptor was not legacy-decoded", peek(i) === 16'ha55a);
        end
    endtask

    initial begin
        rst = 0; wait_cycles(6); rst = 1; wait_cycles(10);
        if (TB_VECTOR_EN) begin
            case (TB_CASE)
                0: run_enabled_case(0, "zero_length");
                1: run_enabled_case(1, "truncated_descriptor");
                2: run_enabled_case(2, "reserved_subop");
                3: run_enabled_case(3, "unknown_param_kind");
                4: run_enabled_case(4, "reserved_record_bits");
                5: run_enabled_case(5, "invalid_signed_range");
                6: run_enabled_case(6, "pointer_extent");
                7: run_enabled_case(7, "v1_shape_violation");
                8: run_enabled_case(8, "illegal_flags");
                9: run_enabled_case(9, "bad_record_kind");
                10: run_enabled_case(10, "prohibited_overlap");
                11: run_enabled_case(11, "softmax_head_id");
                12: run_enabled_case(12, "softmax_noncausal");
                default: begin check("valid TB_CASE", 0); end
            endcase
        end else run_absent_case();
        if (errors == 0) $display("FULL_TOP_DECODE_RESULT %0d 0 %0d", tests, total_cycles);
        else $display("FULL_TOP_DECODE_RESULT %0d %0d %0d", tests, errors, total_cycles);
        $finish;
    end
endmodule
