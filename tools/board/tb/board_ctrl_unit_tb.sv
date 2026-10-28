`timescale 1ns/1ps
// Unit testbench for board_ctrl with a stub decode engine: full-width position path (N11), bounds / sequence / S-frame
// error frames (F06), watchdog (abort pulse, error frame, recovery), reset in the middle of a frame, a step and a response.
// The stub echoes dec_token ^ 5A5Ah as the argmax token; it never looks at board_ctrl internals.
module board_ctrl_unit_tb;
  localparam int PW = 7, VOC = 32000, MS = 128, WD = 300, LAT = 100;
  logic clk = 0, rst = 1;
  always #1 clk = ~clk;
  logic [7:0] rx_data = 0; logic rx_valid = 0;
  logic [7:0] tx_data; logic tx_start;
  logic dec_start, dec_abort; logic [15:0] dec_token; logic [PW-1:0] dec_pos;
  logic dec_done = 0; logic [15:0] dec_token_out = 0;
  logic pt_active, pt_sclk, pt_cs_n, pt_mosi;
  board_ctrl #(.PW(PW), .PT_HALF(2), .TMO(1000), .VOCAB(VOC), .MAX_SEQ(MS), .WDOG(WD)) dut (
    .clk, .rst, .rx_data, .rx_valid, .tx_data, .tx_start, .tx_busy(1'b0), .dec_start, .dec_abort, .dec_token, .dec_pos,
    .dec_done, .dec_token_out, .dec_logit_valid(1'b0), .dec_logit_idx(16'd0), .dec_logit_val(32'sd0),
    .pt_active, .pt_sclk, .pt_cs_n, .pt_mosi, .pt_miso(1'b0));
  integer errors = 0, checks = 0, starts = 0, wr = 0, rd = 0, abort_cycles = 0;
  logic [7:0] q [1024];
  logic [PW-1:0] seen_pos; logic [15:0] seen_tok;
  integer lat_cnt = -1; logic stall = 0;
  always @(posedge clk) begin
    dec_done <= 1'b0;
    if (tx_start) begin q[wr % 1024] <= tx_data; wr <= wr + 1; end
    if (dec_abort) abort_cycles <= abort_cycles + 1;
    if (rst || dec_abort) lat_cnt <= -1;
    else if (dec_start) begin starts <= starts + 1; seen_pos <= dec_pos; seen_tok <= dec_token; lat_cnt <= stall ? -1 : LAT; end
    else if (lat_cnt > 0) lat_cnt <= lat_cnt - 1;
    else if (lat_cnt == 0) begin dec_done <= 1'b1; dec_token_out <= seen_tok ^ 16'h5A5A; lat_cnt <= -1; end
  end
  task automatic send(input [7:0] b);
    @(negedge clk); rx_data = b; rx_valid = 1; @(negedge clk); rx_valid = 0; repeat (2) @(negedge clk);
  endtask
  task automatic chk(input bit ok, input string what);
    checks++; if (!ok) begin errors++; $display("MISMATCH %s", what); end
  endtask
  task automatic wait_bytes(input int n, input int guard);
    int g; g = 0; while (wr - rd < n && g < guard) begin @(posedge clk); g++; end
    chk(wr - rd >= n, "response bytes arrived");
  endtask
  task automatic eb(input [7:0] e, input string what);
    checks++; if (q[rd % 1024] !== e) begin errors++; $display("MISMATCH %s got=%h exp=%h", what, q[rd % 1024], e); end rd++;
  endtask
  task automatic idle_none(input string what);
    repeat (50) @(posedge clk); chk(wr == rd, {"no response after ", what});
  endtask
  task automatic T(input int tok, input int pos);
    send(8'h54); send(tok[7:0]); send(tok[15:8]); send(pos[7:0]); send(8'h00); send(8'h00);
  endtask
  task automatic expect_R(input int tok, input int pos, input string what);
    wait_bytes(11, 2000);
    eb(8'h52, {what, " R tag"}); eb(8'((tok ^ 16'h5A5A) & 255), {what, " token lo"}); eb(8'(((tok ^ 16'h5A5A) >> 8) & 255), {what, " token hi"});
    rd += 8;
    chk(seen_pos === PW'(pos), {what, " dec_pos at the decode port"});
    chk(seen_tok === 16'(tok), {what, " dec_token at the decode port"});
  endtask
  task automatic expect_E(input [7:0] code, input int next, input string what);
    wait_bytes(3, 4000); eb(8'h45, {what, " E tag"}); eb(code, {what, " E code"}); eb(8'(next), {what, " next"});
  endtask
  task automatic reject(input int tok, input int pos, input [7:0] code, input int next, input string what);
    int s0; s0 = starts; T(tok, pos); expect_E(code, next, what); idle_none(what); chk(starts == s0, {what, " ran no decode"});
  endtask
  task automatic doreset;
    @(negedge clk); rst = 1; repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
  endtask
  task automatic ping(input string what);
    send(8'h56); wait_bytes(2, 200); eb(8'h76, {what, " v"}); eb(8'h01, {what, " v1"}); idle_none(what);
  endtask
  initial begin
    repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
    // 1. full-width position walk 0..127 in order (bit 6 of pos reaches the decode port)
    for (int p = 0; p < MS; p++) begin T((p * 251 + 7) % VOC, p); expect_R((p * 251 + 7) % VOC, p, $sformatf("walk pos %0d", p)); end
    chk(starts == MS, "128 steps started");
    // 2. bounds
    reject(1, 128, 8'h02, 128, "pos == MAX_SEQ");
    reject(1, 255, 8'h02, 128, "pos 255");
    reject(1, 192, 8'h02, 128, "pos 192");
    reject(32000, 0, 8'h01, 128, "token == VOCAB");
    reject(65535, 0, 8'h01, 128, "token 65535");
    reject(32000, 200, 8'h01, 128, "token and pos bad -> token code first");
    T(31999, 0); expect_R(31999, 0, "token VOCAB-1");
    // 3. sequence rule
    send(8'h58); wait_bytes(2, 200); eb(8'h78, "x tag"); eb(8'h01, "x ver");
    reject(1, 1, 8'h03, 0, "gap after X");
    T(5, 0); expect_R(5, 0, "pos 0 after X");
    reject(5, 2, 8'h03, 1, "gap pos 2, next 1");
    T(6, 1); expect_R(6, 1, "pos 1");
    T(7, 0); expect_R(7, 0, "re-run pos 0");
    reject(5, 3, 8'h03, 2, "gap pos 3, next 2 (re-run did not move next)");
    // 4. S frame argument errors
    send(8'h53); send(8'd9); send(8'd1); expect_E(8'h05, 2, "S ntx 9"); idle_none("S ntx 9");
    send(8'h53); send(8'd1); send(8'd9); expect_E(8'h05, 2, "S nrx 9"); idle_none("S nrx 9");
    // 5. watchdog: the stub never finishes; abort pulse, E 04, next unchanged, then a normal step works
    stall = 1; abort_cycles = 0; T(9, 2);
    expect_E(8'h04, 2, "watchdog");
    chk(abort_cycles >= 16 && abort_cycles <= 20, "dec_abort held 16..20 cycles");
    stall = 0; idle_none("watchdog");
    T(9, 2); expect_R(9, 2, "step after watchdog");
    T(10, 3); expect_R(10, 3, "step under the watchdog limit");     // LAT=100 < WD=300
    // 6. reset in the middle of a frame, of a step, of the response
    send(8'h54); send(8'd1); send(8'd0); doreset; idle_none("reset mid frame");
    ping("after reset mid frame");
    reject(1, 1, 8'h03, 0, "next cleared by reset");
    T(11, 0);
    repeat (30) @(posedge clk); doreset;                   // reset while the step runs
    idle_none("reset mid step");
    ping("after reset mid step");
    T(12, 0); expect_R(12, 0, "step after reset mid step");
    T(13, 1); wait_bytes(1, 2000); doreset;                // reset while the response is being sent
    rd = wr; ping("after reset mid response");
    T(14, 0); expect_R(14, 0, "step after reset mid response");
    $display("SUMMARY errors=%0d checks=%0d starts=%0d", errors, checks, starts);
    $finish;
  end
  initial begin #50000000; $display("MISMATCH global timeout"); $display("SUMMARY errors=%0d checks=%0d starts=%0d", errors + 1, checks, starts); $finish; end
endmodule
