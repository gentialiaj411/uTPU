// Self-test of qspi_flash_model.sv (EDh DDR Quad I/O Read, 6Bh latency by LC, error checks). Hand-written, cycle-explicit host
// stimulus; no reader is involved. Expected values are derived from the S25FL128S datasheet Rev *T (Fig. 94 p.105, Table 27 p.54,
// Table 15 p.44) and written here independently of the model. Run through tools/decode/run_flash_model_selftest.py.
// Time grid: SCK period 20 ns (50 MHz). DDR host data changes 5 ns after one SCK edge, i.e. 5 ns before the next (tSU/tHD margin 5 ns).
`timescale 1ns/1ps

module ddr_env #(parameter bit SPARSE = 1'b0, parameter bit QUAD = 1'b1, parameter bit [1:0] LC = 2'b00, parameter bit REQ = 1'b1,
                 parameter int SIZE = 4096) ();
    logic sck = 0, cs_n = 1; logic [3:0] host_o = 0, host_oe = 0;
    wire [3:0] f_o, f_oe, bus;
    qspi_flash_model #(.IMG("build/flash_selftest_img.hex"), .SIZE(SIZE), .SPARSE(SPARSE), .QUAD_INIT(QUAD), .LC_INIT(LC), .REQUIRE_LC00(REQ), .FATAL_ON_ERROR(1'b0),
                       .TCSS(3.0), .TCSH(3.0), .TCS(10.0)) fl (.sck, .cs_n, .host_o, .host_oe, .flash_o(f_o), .flash_oe(f_oe));
    for (genvar i = 0; i < 4; i++) begin : g_bus
        assign bus[i] = host_oe[i] ? host_o[i] : (f_oe[i] ? f_o[i] : 1'bz);
    end
    logic [7:0] img [SIZE];
    initial if (!SPARSE) $readmemh("build/flash_selftest_img.hex", img);
    function automatic logic [7:0] exp_byte(input logic [31:0] a);   // independent of the model's sparse_byte
        longint unsigned h;
        if (SPARSE) begin
            h = longint'(a) ^ (longint'(a) >> 13); h = (h * 64'd2246822507) & 64'hFFFF_FFFF; h = h ^ (h >> 16);
            return 8'(h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24));
        end else return (a < SIZE) ? img[a] : 8'hFF;
    endfunction
    logic [3:0] rxn [0:511]; int nrx = 0;
    // knobs (set by the test before a call; -1 = off)
    int k_setup = -1, k_hold = -1, k_drv_dummy = -1, k_drv_data = -1, k_abort_nib = -1, k_abort_dummy = -1;
    task automatic cs_low(); #20 cs_n = 0; #5; endtask
    task automatic cs_high(); #5 cs_n = 1; host_oe = 4'h0; #1; endtask
    // EDh transaction. dummy_h = SCK cycles of dummy the host provides, samp = ns after each data edge at which the host samples.
    task automatic ddr_read(input logic [7:0] op, input logic [23:0] addr, input logic [7:0] mode, input int dummy_h, input int nbytes, input real samp);
        logic [3:0] nb [8]; bit ab = 0;
        nb[0] = addr[23:20]; nb[1] = addr[19:16]; nb[2] = addr[15:12]; nb[3] = addr[11:8]; nb[4] = addr[7:4]; nb[5] = addr[3:0];
        nb[6] = mode[7:4]; nb[7] = mode[3:0];
        nrx = 0;
        cs_low();
        for (int i = 7; i >= 0; i--) begin host_oe = 4'h1; host_o = {3'b000, op[i]}; #5 sck = 1; #5 sck = 0; end   // 8 SDR cycles on IO0
        for (int k = 0; k < 8; k++) begin                                          // cycles 8..11: one nibble per edge, rising first
            if (!ab && k == k_abort_nib) begin #5 host_oe = 0; cs_high(); ab = 1; end
            if (ab) ;
            else if (k == k_setup) begin #9 host_oe = 4'hF; host_o = nb[k]; #1 sck = ~sck; end
            else begin #5 host_oe = 4'hF; host_o = nb[k]; #5 sck = ~sck; end
            if (!ab && k == k_hold) #1 host_o = ~host_o;
        end
        for (int e = 0; e < 2 * dummy_h; e++) begin                                // dummy: host releases the bus
            if (!ab && e == k_abort_dummy) begin host_oe = 0; cs_high(); ab = 1; end
            if (!ab) begin #5 host_oe = (e == k_drv_dummy) ? 4'hF : 4'h0; host_o = 4'h5; #5 sck = ~sck; end
        end
        if (!ab) host_oe = 0;
        for (int e = 0; e < 2 * nbytes && !ab; e++) begin
            if (e == 0) #10; else #(10.0 - samp);
            if (e == k_drv_data) begin host_oe = 4'hF; host_o = 4'h9; end
            sck = ~sck; if (e == k_drv_data) #0.5 host_oe = 0;
            #(samp) rxn[e] = bus; nrx = e + 1;
        end
        if (!ab) begin #(10.0 - samp); cs_high(); end                                                 // sck is low after an even number of data edges
    endtask
    // 6Bh transaction (SDR, all on IO0, data nibbles on the falling edge launch). dummy_h = SCK cycles the host provides.
    task automatic sdr_read(input logic [23:0] addr, input int dummy_h, input int nbytes);
        logic [31:0] frame; frame = {8'h6B, addr};
        nrx = 0; cs_low();
        for (int i = 31; i >= 0; i--) begin
            host_oe = 4'h1; host_o = {3'b000, frame[i]}; #5 sck = 1;
            if (i == 0) begin #2 host_oe = 0; #3; end else #5;     // release the bus before the falling edge that launches data
            sck = 0;
        end
        for (int d = 0; d < dummy_h; d++) begin #5 sck = 1; #5 sck = 0; end
        for (int e = 0; e < 2 * nbytes; e++) begin #8 rxn[e] = bus; nrx = e + 1; #2 sck = 1; #10 sck = 0; end
        #5 cs_high();
    endtask
    function automatic int mism(input logic [31:0] addr, input int nbytes);      // nibbles that differ from the expected bytes
        int n = 0; logic [7:0] b;
        for (int i = 0; i < nbytes; i++) begin
            b = exp_byte((addr + i) & 32'hFFFFFF);
            if (rxn[2*i] !== b[7:4]) n++;
            if (rxn[2*i+1] !== b[3:0]) n++;
        end
        return n;
    endfunction
    function automatic int nx(input int nnib);                                    // nibbles that contain X
        int n = 0; for (int i = 0; i < nnib; i++) if ((^rxn[i]) === 1'bx) n++; return n;
    endfunction
endmodule

module test_flash_model_ddr;
    int checks = 0, fails = 0, errors_expected = 0, errors_seen = 0;
    ddr_env #(.SIZE(4096)) a ();                         // LC=00 QUAD=1 image
    ddr_env #(.SPARSE(1'b1)) b ();                                                             // LC=00, full 16 MiB hash space
    ddr_env #(.LC(2'b00), .REQ(1'b1), .SIZE(4096)) c (); // LC=00, project policy on
    ddr_env #(.LC(2'b11), .REQ(1'b1), .SIZE(4096)) c11 (); // legacy LC=11 policy negative
    ddr_env #(.LC(2'b00), .REQ(1'b0), .SIZE(4096)) d (); // LC=00, datasheet table rows
    ddr_env #(.LC(2'b01), .REQ(1'b0), .SIZE(4096)) e1 ();
    ddr_env #(.LC(2'b10), .REQ(1'b0), .SIZE(4096)) e2 ();
    ddr_env #(.QUAD(1'b0), .SIZE(4096)) q0 ();            // QUAD=0
    task automatic chk(input bit ok, input string name);
        checks++; if (!ok) begin fails++; $display("FAIL %s", name); end else $display("ok   %s", name);
    endtask
    task automatic experr(input int got, input int exp, input string name);       // got/exp = model errors raised by this step
        errors_expected += exp; chk(got == exp, {name, $sformatf(" (errors %0d, expected %0d)", got, exp)});
    endtask
    int e0;
    initial begin
        // ---- positive reads, LC=00, QUAD=1, datasheet Fig. 94: 6 dummy cycles ----
        e0 = a.fl.err_total;
        a.ddr_read(8'hED, 24'h000000, 8'h00, 6, 8, 8.0);     chk(a.mism(0, 8) == 0 && a.nrx == 16, "EDh addr 0 (8 bytes)");
        #30 a.ddr_read(8'hED, 24'h0007F1, 8'h00, 6, 17, 8.0); chk(a.mism(32'h7F1, 17) == 0, "EDh unaligned start 0x7F1 (17 bytes)");
        #30 a.ddr_read(8'hED, 24'h000FF0, 8'h00, 6, 16, 8.0); chk(a.mism(32'hFF0, 16) == 0, "EDh ending exactly at the image end 0xFFF");
        #30 a.ddr_read(8'hED, 24'h000123, 8'h00, 6, 6, 6.6);  chk(a.mism(32'h123, 6) == 0, "EDh host sample 0.1 ns after tV(max)");
        experr(a.fl.err_total - e0, 0, "no errors on legal EDh reads");
        e0 = a.fl.over_read;
        #30 a.ddr_read(8'hED, 24'h000FFE, 8'h00, 6, 4, 8.0);
        chk(a.mism(32'hFFE, 4) == 0 && a.fl.over_read - e0 == 2, "EDh over the image end reads 0xFF (over_read counts 2)");
        e0 = a.fl.err_total;
        // ---- sampling inside the invalid window reads X ----
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 8, 5.5);   chk(a.nx(16) >= 15 && a.mism(32'h40, 8) > 0, "host sampling at tV(max)-1.0 ns reads X");
        experr(a.fl.err_total - e0, 0, "early sampling is a host-side failure, not a model error");
        // ---- wrap at the end of the array (sparse model, full 24-bit space) ----
        e0 = b.fl.err_total;
        b.ddr_read(8'hED, 24'hFFFFFC, 8'h00, 6, 12, 8.0);      chk(b.mism(32'hFFFFFC, 12) == 0, "EDh wraps 0xFFFFFF -> 0x000000");
        #30 b.ddr_read(8'hED, 24'h280000, 8'h00, 6, 8, 8.0);   chk(b.mism(32'h280000, 8) == 0, "EDh sparse 0x280000");
        #30 b.ddr_read(8'hED, 24'h8001A5, 8'h00, 6, 8, 8.0);   chk(b.mism(32'h8001A5, 8) == 0, "EDh sparse 0x8001A5 (high address bits)");
        experr(b.fl.err_total - e0, 0, "no errors on sparse EDh reads");
        // ---- 6Bh dummy count comes from LC (Table 26) ----
        e0 = a.fl.err_total;
        #30 a.sdr_read(24'h000040, 8, 8);   chk(a.mism(32'h40, 8) == 0, "6Bh at LC=00 uses 8 dummy");
        experr(a.fl.err_total - e0, 0, "6Bh at LC=00 no errors");
        d.sdr_read(24'h000040, 8, 8);       chk(d.mism(32'h40, 8) == 0, "6Bh at LC=00 uses 8 dummy");
        #30 d.sdr_read(24'h000040, 0, 8);   chk(d.mism(32'h40, 8) > 0 && d.nx(16) > 0, "6Bh at LC=00 with a 0-dummy host is detectable (no data)");
        // ---- LC table rows for EDh (Table 27: 00 -> 6, 01 -> 7, 10 -> 8) ----
        d.ddr_read(8'hED, 24'h000300, 8'h00, 6, 8, 8.0);   chk(d.mism(32'h300, 8) == 0, "EDh at LC=00 uses 6 dummy (REQUIRE_LC00=0)");
        #30 d.ddr_read(8'hED, 24'h000300, 8'h00, 3, 8, 8.0); chk(d.mism(32'h300, 8) > 0, "EDh at LC=00 with a 3-dummy host is detectable");
        e1.ddr_read(8'hED, 24'h000300, 8'h00, 7, 8, 8.0);  chk(e1.mism(32'h300, 8) == 0, "EDh at LC=01 uses 7 dummy");
        e2.ddr_read(8'hED, 24'h000300, 8'h00, 8, 8, 8.0);  chk(e2.mism(32'h300, 8) == 0, "EDh at LC=10 uses 8 dummy");
        experr(d.fl.err_total + e1.fl.err_total + e2.fl.err_total, 0, "no errors on the LC table rows with REQUIRE_LC00=0");
        // ---- negative checks: each must fire exactly as expected ----
        e0 = c.fl.err_total;
        c.ddr_read(8'hED, 24'h000300, 8'h00, 6, 8, 8.0);
        experr(c.fl.err_total - e0, 0, "EDh at LC=00 accepted (project policy)"); chk(c.fl.err_lc == 0, "  err_lc counter = 0");
        e0 = c11.fl.err_total;
        c11.ddr_read(8'hED, 24'h000300, 8'h00, 3, 8, 8.0);
        experr(c11.fl.err_total - e0, 1, "EDh at LC=11 rejected (LC != 00)"); chk(c11.fl.err_lc == 1, "  err_lc counter = 1");
        e0 = q0.fl.err_total;
        q0.ddr_read(8'hED, 24'h000300, 8'h00, 6, 8, 8.0);
        experr(q0.fl.err_total - e0, 1, "EDh with QUAD=0 rejected"); chk(q0.fl.err_quad == 1, "  err_quad counter = 1");
        e0 = a.fl.err_total; a.k_drv_dummy = 4;
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 4, 8.0); a.k_drv_dummy = -1;
        experr(a.fl.err_total - e0, 1, "host drives IO in the dummy phase (contention)");
        e0 = a.fl.err_total; a.k_drv_data = 2;
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 4, 8.0); a.k_drv_data = -1;
        experr(a.fl.err_total - e0, 2, "host drives IO while the flash drives data (contention: level check + edge check)");
        chk(a.fl.err_contend == 3, "  err_contend counter = 3");
        e0 = a.fl.err_total; a.k_setup = 3;
        #30 a.ddr_read(8'hED, 24'h000A5C, 8'h00, 6, 4, 8.0); a.k_setup = -1;
        experr(a.fl.err_total - e0, 1, "address nibble changes 1 ns before the edge (tSU)"); chk(a.fl.err_setup == 1, "  err_setup counter = 1");
        e0 = a.fl.err_total; a.k_hold = 4;
        #30 a.ddr_read(8'hED, 24'h000A5C, 8'h00, 6, 4, 8.0); a.k_hold = -1;
        experr(a.fl.err_total - e0, 1, "address nibble changes 1 ns after the edge (tHD)"); chk(a.fl.err_hold == 1, "  err_hold counter = 1");
        e0 = a.fl.err_total;
        #30 a.ddr_read(8'hED, 24'h000040, 8'hA5, 6, 4, 8.0);
        experr(a.fl.err_total - e0, 1, "mode byte A5h (complementary nibbles = continuous mode) flagged"); chk(a.fl.err_cont == 1, "  err_cont counter = 1");
        e0 = a.fl.err_total;
        #30 a.ddr_read(8'hED, 24'h000040, 8'hA0, 6, 4, 8.0);
        experr(a.fl.err_total - e0, 0, "mode byte A0h is not complementary: not continuous mode (datasheet p.104)");
        e0 = a.fl.err_total; a.k_abort_nib = 4;
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 4, 8.0); a.k_abort_nib = -1;
        experr(a.fl.err_total - e0, 1, "CS# raised mid-address"); chk(a.fl.err_csmid == 1, "  err_csmid counter = 1");
        e0 = a.fl.err_total; a.k_abort_dummy = 2;
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 4, 8.0); a.k_abort_dummy = -1;
        experr(a.fl.err_total - e0, 1, "CS# raised in the dummy phase"); chk(a.fl.err_csmid == 2, "  err_csmid counter = 2");
        e0 = a.fl.err_total;
        #30 a.ddr_read(8'hED, 24'h000040, 8'h00, 6, 8, 8.0); chk(a.mism(32'h40, 8) == 0, "model recovers: legal EDh after the aborted ones");
        experr(a.fl.err_total - e0, 0, "no errors on the recovery read");
        errors_seen = a.fl.err_total + b.fl.err_total + c.fl.err_total + c11.fl.err_total + d.fl.err_total + e1.fl.err_total + e2.fl.err_total + q0.fl.err_total;
        chk(errors_seen == errors_expected, $sformatf("sticky counters sum to the expected total (%0d vs %0d)", errors_seen, errors_expected));
        $display("SUMMARY checks=%0d errors_expected=%0d errors_seen=%0d fails=%0d", checks, errors_expected, errors_seen, fails);
        $finish;
    end
    initial begin #50_000_000; $display("FAIL watchdog"); $display("SUMMARY checks=%0d errors_expected=%0d errors_seen=%0d fails=%0d", checks, errors_expected, errors_seen, fails + 1); $finish; end
endmodule
