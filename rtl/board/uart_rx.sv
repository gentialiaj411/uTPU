// 8N1 UART receiver. CPB = clock cycles per bit. valid is a one-cycle pulse; a bad stop bit drops the byte.
module uart_rx #(parameter int CPB = 434) (
    input  logic       clk, rst, rxd,
    output logic [7:0] data,
    output logic       valid
);
    localparam int CW = $clog2(CPB + 1) + 1;
    logic r0, r1;
    logic [CW-1:0] cnt;
    logic [3:0] nbit;
    logic [7:0] sh;
    typedef enum logic [1:0] {S_IDLE, S_START, S_DATA, S_STOP} st_t;
    st_t st;
    always_ff @(posedge clk) begin
        r0 <= rxd; r1 <= r0; valid <= 1'b0;
        if (rst) begin st <= S_IDLE; cnt <= '0; nbit <= '0; end
        else case (st)
            S_IDLE:  if (!r1) begin st <= S_START; cnt <= CW'(CPB / 2); end
            S_START: if (cnt == 0) begin
                         if (!r1) begin st <= S_DATA; cnt <= CW'(CPB - 1); nbit <= '0; end else st <= S_IDLE;
                     end else cnt <= cnt - 1'b1;
            S_DATA:  if (cnt == 0) begin
                         sh <= {r1, sh[7:1]}; cnt <= CW'(CPB - 1); nbit <= nbit + 1'b1;
                         if (nbit == 4'd7) st <= S_STOP;
                     end else cnt <= cnt - 1'b1;
            S_STOP:  if (cnt == 0) begin
                         if (r1) begin data <= sh; valid <= 1'b1; end
                         st <= S_IDLE;
                     end else cnt <= cnt - 1'b1;
        endcase
    end
endmodule
