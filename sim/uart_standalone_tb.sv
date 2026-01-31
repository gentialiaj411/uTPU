//=============================================================================
// UART Standalone Testbench - Fixed version
// Tests the corrected UART receiver timing
//=============================================================================

`timescale 1ns/1ps

module uart_standalone_tb;

    // Parameters matching your design
    localparam integer INPUT_CLK = 100_000_000;  // 100 MHz
    localparam integer UART_BAUD = 115200;
    localparam integer OVERSAMPLE = 16;
    localparam integer UART_BITS = 8;
    
    // Derived timing
    localparam integer UART_CLK_OS = UART_BAUD * OVERSAMPLE;
    localparam integer DIVIDER = INPUT_CLK / UART_CLK_OS;  // = 54
    localparam real CLK_PERIOD_NS = 10.0;  // 100 MHz = 10ns period
    localparam real BIT_PERIOD_NS = 1e9 / UART_BAUD;  // ~8680.56 ns
    
    // Signals
    logic clk = 0;
    logic rst = 1;  // Start in reset (active high for uart module directly)
    logic rx = 1;  // UART idle = high
    logic tx;
    logic rx_valid;
    logic [7:0] rx_result;
    logic tx_busy;
    
    // Test state
    logic [7:0] expected_byte;
    logic byte_received;
    logic [7:0] received_byte;
    int tests_passed = 0;
    int tests_failed = 0;
    
    // Clock generation
    always #(CLK_PERIOD_NS/2) clk = ~clk;
    
    // DUT - UART module
    uart #(
        .UART_BITS_TRANSFERED(UART_BITS),
        .INPUT_CLK(INPUT_CLK),
        .UART_CLK(UART_BAUD),
        .OVERSAMPLE(OVERSAMPLE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .tx_start(1'b0),
        .rx(rx),
        .rx_valid(rx_valid),
        .tx(tx),
        .tx_message(8'h00),
        .rx_result(rx_result),
        .tx_busy(tx_busy)
    );
    
    //=========================================================================
    // Monitor: Capture rx_valid pulses asynchronously
    //=========================================================================
    always @(posedge clk) begin
        if (rx_valid) begin
            byte_received <= 1'b1;
            received_byte <= rx_result;
            $display("[%0t] RX: Received byte 0x%02X", $time, rx_result);
        end
    end
    
    //=========================================================================
    // Task: Send a byte via UART (simulating a PC sending to FPGA)
    //=========================================================================
    task automatic uart_send_byte(input logic [7:0] data);
        int i;
        
        // Clear receive flag
        byte_received = 1'b0;
        expected_byte = data;
        
        $display("[%0t] TX: Sending byte 0x%02X (%08b)", $time, data, data);
        
        // Start bit (low)
        rx = 1'b0;
        #(BIT_PERIOD_NS);
        
        // Data bits (LSB first)
        for (i = 0; i < 8; i++) begin
            rx = data[i];
            #(BIT_PERIOD_NS);
        end
        
        // Stop bit (high)
        rx = 1'b1;
        #(BIT_PERIOD_NS);
        
        // Extra idle time for processing
        #(BIT_PERIOD_NS * 2);
    endtask
    
    //=========================================================================
    // Task: Check if byte was received correctly
    //=========================================================================
    task automatic check_received(input logic [7:0] expected);
        // Wait a bit more for any pipeline delays
        #(CLK_PERIOD_NS * 100);
        
        if (!byte_received) begin
            $display("[%0t] FAIL: No byte received (expected 0x%02X)", $time, expected);
            tests_failed++;
        end else if (received_byte !== expected) begin
            $display("[%0t] FAIL: Received 0x%02X, expected 0x%02X", $time, received_byte, expected);
            tests_failed++;
        end else begin
            $display("[%0t] PASS: Byte 0x%02X received correctly", $time, expected);
            tests_passed++;
        end
        
        // Reset for next test
        byte_received = 1'b0;
    endtask
    
    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("==============================================");
        $display("UART Standalone Testbench (Fixed)");
        $display("Baud rate: %0d, Oversampling: %0dx", UART_BAUD, OVERSAMPLE);
        $display("Clock divider: %0d (actual baud: %0d)", DIVIDER, INPUT_CLK / DIVIDER / OVERSAMPLE);
        $display("Bit period: %0.2f ns", BIT_PERIOD_NS);
        $display("==============================================");
        
        // Initialize
        byte_received = 0;
        received_byte = 0;
        
        // Reset sequence
        rst = 1;
        rx = 1;
        #200;
        rst = 0;
        #2000;
        
        $display("\n--- Test 1: Send 0x55 (alternating bits: 01010101) ---");
        uart_send_byte(8'h55);
        check_received(8'h55);
        
        $display("\n--- Test 2: Send 0xAA (alternating bits: 10101010) ---");
        uart_send_byte(8'hAA);
        check_received(8'hAA);
        
        $display("\n--- Test 3: Send 0x00 (all zeros) ---");
        uart_send_byte(8'h00);
        check_received(8'h00);
        
        $display("\n--- Test 4: Send 0xFF (all ones) ---");
        uart_send_byte(8'hFF);
        check_received(8'hFF);
        
        $display("\n--- Test 5: Send 0xA5 (10100101) ---");
        uart_send_byte(8'hA5);
        check_received(8'hA5);
        
        $display("\n--- Test 6: Send 0x5A (01011010) ---");
        uart_send_byte(8'h5A);
        check_received(8'h5A);
        
        $display("\n--- Test 7: Send 0x12 ---");
        uart_send_byte(8'h12);
        check_received(8'h12);
        
        $display("\n--- Test 8: Send 0x34 ---");
        uart_send_byte(8'h34);
        check_received(8'h34);
        
        $display("\n--- Test 9: Send 0x78 ---");
        uart_send_byte(8'h78);
        check_received(8'h78);
        
        $display("\n--- Test 10: Send 0xDE ---");
        uart_send_byte(8'hDE);
        check_received(8'hDE);
        
        // Summary
        #1000;
        $display("\n==============================================");
        $display("TEST SUMMARY: %0d passed, %0d failed", tests_passed, tests_failed);
        if (tests_failed == 0)
            $display("*** ALL TESTS PASSED - UART IS WORKING! ***");
        else
            $display("*** SOME TESTS FAILED ***");
        $display("==============================================");
        
        #1000;
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #50_000_000;  // 50ms timeout
        $display("ERROR: Global timeout reached!");
        $finish;
    end

endmodule
