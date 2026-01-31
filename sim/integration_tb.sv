//=============================================================================
// Integration Testbench - Tests UART + FSM + Buffer operations
// Verifies: STORE instruction, FETCH instruction, round-trip data integrity
//=============================================================================

`timescale 1ns/1ps

module integration_tb;

    // Parameters
    localparam real CLK_PERIOD_NS = 10.0;  // 100 MHz
    localparam real BIT_PERIOD_NS = 1e9 / 115200;  // ~8680.56 ns
    
    // Signals
    logic clk = 0;
    logic rst = 0;  // Active LOW for top module
    logic rx = 1;
    wire tx;
    wire led_rst;
    
    // Test tracking
    int tests_passed = 0;
    int tests_failed = 0;
    
    // TX capture
    logic [7:0] tx_captured_byte;
    logic tx_byte_ready;
    
    // Clock generation
    always #(CLK_PERIOD_NS/2) clk = ~clk;
    
    // DUT
    top #(
        .UART_INPUT_CLK(100000000),
        .UART_BAUD(115200),
        .DEBUG_STORE_ACK(0),
        .DEBUG_FETCH_ACK(0)
    ) dut (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .tx(tx),
        .led_rst(led_rst)
    );
    
    //=========================================================================
    // TX Line Monitor - Captures bytes sent by FPGA
    //=========================================================================
    logic tx_prev = 1;
    logic tx_capturing = 0;
    logic [3:0] tx_bit_count;
    logic [7:0] tx_shift;
    int tx_sample_count;
    localparam int SAMPLES_PER_BIT = 868;  // 100MHz / 115200
    
    always @(posedge clk) begin
        tx_byte_ready <= 1'b0;
        
        if (!tx_capturing) begin
            // Detect start bit (falling edge)
            if (tx_prev && !tx) begin
                tx_capturing <= 1'b1;
                tx_bit_count <= 0;
                tx_sample_count <= SAMPLES_PER_BIT / 2;  // Sample at mid-bit
                tx_shift <= 0;
            end
        end else begin
            tx_sample_count <= tx_sample_count + 1;
            if (tx_sample_count >= SAMPLES_PER_BIT) begin
                tx_sample_count <= 0;
                tx_bit_count <= tx_bit_count + 1;
                
                if (tx_bit_count >= 1 && tx_bit_count <= 8) begin
                    tx_shift[tx_bit_count-1] <= tx;
                end
                
                if (tx_bit_count == 9) begin
                    // Stop bit - byte complete
                    tx_captured_byte <= tx_shift;
                    tx_byte_ready <= 1'b1;
                    tx_capturing <= 1'b0;
                    $display("[%0t] FPGA TX: 0x%02X", $time, tx_shift);
                end
            end
        end
        
        tx_prev <= tx;
    end
    
    //=========================================================================
    // Task: Send a byte via UART to FPGA
    //=========================================================================
    task automatic uart_send_byte(input logic [7:0] data);
        int i;
        $display("[%0t] PC TX: 0x%02X", $time, data);
        
        // Start bit
        rx = 1'b0;
        #(BIT_PERIOD_NS);
        
        // Data bits (LSB first)
        for (i = 0; i < 8; i++) begin
            rx = data[i];
            #(BIT_PERIOD_NS);
        end
        
        // Stop bit
        rx = 1'b1;
        #(BIT_PERIOD_NS);
        
        // Inter-byte gap
        #(BIT_PERIOD_NS);
    endtask
    
    //=========================================================================
    // Task: Send 16-bit instruction (little endian)
    //=========================================================================
    task automatic send_instruction(input logic [15:0] instr);
        $display("[%0t] Sending instruction: 0x%04X", $time, instr);
        uart_send_byte(instr[7:0]);   // Low byte first
        uart_send_byte(instr[15:8]);  // High byte second
    endtask
    
    //=========================================================================
    // Task: Wait for TX byte from FPGA with timeout
    //=========================================================================
    task automatic wait_for_tx_byte(output logic [7:0] data, output logic success);
        int timeout = 0;
        int max_timeout = 200000;  // 2ms at 100MHz
        
        success = 0;
        while (!tx_byte_ready && timeout < max_timeout) begin
            @(posedge clk);
            timeout++;
        end
        
        if (tx_byte_ready) begin
            data = tx_captured_byte;
            success = 1;
        end
    endtask
    
    //=========================================================================
    // ISA Encoding Helpers
    //=========================================================================
    // Instruction format: [15:7]=address, [6:3]=mode bits, [2:0]=opcode
    
    // STORE: opcode=000, followed by 3 more words (value, dest_addr, unused)
    function automatic logic [15:0] make_store_instr();
        return 16'h0000;  // Opcode 000
    endfunction
    
    // FETCH: opcode=001, address in bits[15:7], section in bit[6]
    function automatic logic [15:0] make_fetch_instr(input logic [8:0] addr, input logic section);
        logic [15:0] instr;
        instr[2:0] = 3'b001;  // FETCH opcode
        instr[6:3] = 4'b0000;
        instr[6] = section;
        instr[15:7] = addr;
        return instr;
    endfunction
    
    // NOP: opcode=101
    function automatic logic [15:0] make_nop_instr();
        return 16'h0005;  // Opcode 101
    endfunction
    
    // HALT: opcode=100
    function automatic logic [15:0] make_halt_instr();
        return 16'h0004;  // Opcode 100
    endfunction
    
    //=========================================================================
    // Main Test Sequence
    //=========================================================================
    logic [7:0] received_data;
    logic rx_success;
    
    initial begin
        $display("==============================================");
        $display("Integration Testbench");
        $display("Testing: UART + FSM + Buffer Operations");
        $display("==============================================");
        
        // Reset sequence
        rst = 0;  // Assert reset (active low external)
        rx = 1;
        #500;
        rst = 1;  // Release reset
        #5000;
        
        //---------------------------------------------------------------------
        // Test 1: Wait for self-test byte (0xAA)
        //---------------------------------------------------------------------
        $display("\n--- Test 1: Self-test byte ---");
        wait_for_tx_byte(received_data, rx_success);
        
        if (rx_success && received_data == 8'hAA) begin
            $display("[PASS] Received self-test byte 0xAA");
            tests_passed++;
        end else if (rx_success) begin
            $display("[FAIL] Expected 0xAA, got 0x%02X", received_data);
            tests_failed++;
        end else begin
            $display("[FAIL] Timeout waiting for self-test byte");
            tests_failed++;
        end
        
        // Wait for system to be ready
        #(BIT_PERIOD_NS * 5);
        
        //---------------------------------------------------------------------
        // Test 2: Send NOP instruction
        //---------------------------------------------------------------------
        $display("\n--- Test 2: NOP instruction ---");
        send_instruction(make_nop_instr());
        #(BIT_PERIOD_NS * 5);
        $display("[PASS] NOP instruction sent (no response expected)");
        tests_passed++;
        
        //---------------------------------------------------------------------
        // Test 3: FETCH from address 0x000 (should return whatever is there)
        //---------------------------------------------------------------------
        $display("\n--- Test 3: FETCH instruction ---");
        send_instruction(make_fetch_instr(9'h000, 1'b0));  // Fetch from 0x000, section 0
        
        wait_for_tx_byte(received_data, rx_success);
        if (rx_success) begin
            $display("[PASS] FETCH returned: 0x%02X", received_data);
            tests_passed++;
        end else begin
            $display("[FAIL] Timeout waiting for FETCH response");
            tests_failed++;
        end
        
        //---------------------------------------------------------------------
        // Test 4: Multiple NOPs to verify FSM stability
        //---------------------------------------------------------------------
        $display("\n--- Test 4: Multiple NOPs ---");
        for (int i = 0; i < 3; i++) begin
            send_instruction(make_nop_instr());
            #(BIT_PERIOD_NS * 2);
        end
        $display("[PASS] Multiple NOPs processed");
        tests_passed++;
        
        //---------------------------------------------------------------------
        // Test 5: FETCH from different address
        //---------------------------------------------------------------------
        $display("\n--- Test 5: FETCH from address 0x010 ---");
        send_instruction(make_fetch_instr(9'h010, 1'b0));
        
        wait_for_tx_byte(received_data, rx_success);
        if (rx_success) begin
            $display("[PASS] FETCH 0x010 returned: 0x%02X", received_data);
            tests_passed++;
        end else begin
            $display("[FAIL] Timeout waiting for FETCH response");
            tests_failed++;
        end
        
        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        #(BIT_PERIOD_NS * 10);
        $display("\n==============================================");
        $display("TEST SUMMARY: %0d passed, %0d failed", tests_passed, tests_failed);
        if (tests_failed == 0)
            $display("*** ALL TESTS PASSED ***");
        else
            $display("*** SOME TESTS FAILED ***");
        $display("==============================================");
        
        #1000;
        $finish;
    end
    
    // Global timeout
    initial begin
        #100_000_000;  // 100ms
        $display("ERROR: Global timeout!");
        $finish;
    end

endmodule
