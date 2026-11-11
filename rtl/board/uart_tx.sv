// 8N1 UART transmitter. Pulse start for one cycle while busy is low.
module uart_tx #(parameter int CPB = 434) (
    input  logic       clk, rst, start,
    input  logic [7:0] data,
    output logic       txd, busy
);
    localparam int CW = $clog2(CPB + 1) + 1;
    logic [9:0] sh;
    logic [3:0] nbit;
    logic [CW-1:0] cnt;
    assign busy = (nbit != 0);
    always_ff @(posedge clk) begin
        if (rst) begin txd <= 1'b1; nbit <= '0; cnt <= '0; end
        else if (nbit == 0) begin
            txd <= 1'b1;
            if (start) begin sh <= {1'b1, data, 1'b0}; nbit <= 4'd10; cnt <= CW'(CPB - 1); end
        end else begin
            txd <= sh[0];
            if (cnt == 0) begin
                sh <= {1'b1, sh[9:1]}; nbit <= nbit - 1'b1; cnt <= CW'(CPB - 1);
            end else cnt <= cnt - 1'b1;
        end
    end
endmodule
