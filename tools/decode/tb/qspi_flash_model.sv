// Behavioral model of the Boolean board's QSPI flash (Spansion/Infineon S25FL128S) for the read path
// only. Written from the S25FL128S datasheet (rev 05, sec. 10.4.4 Quad Output Read, tables 8.10/8.12,
// AC characteristics), not from rtl/decode/qspi_reader.sv.
//
//  * SPI mode 0. Instruction 6Bh (3-byte address) or 6Ch (4-byte address) on IO0, then DUMMY
//    clocks (8 for latency code 00 at <= 104 MHz), then 4 bits per SCLK on IO[3:0] on the falling
//    edge, high nibble first. The address auto-increments. Data is driven X from tHO after a falling
//    edge until tV after it, so sampling inside that window (or during the dummy clocks, where the
//    bus is Hi-Z) yields X/Z.
//  * $fatal on: wrong instruction; address beyond the image; CS# raised in the middle of a byte of the
//    instruction/address/dummy phase; CS# edge while SCLK is high; CS# high time < tCS; host driving the
//    bus while the flash drives data; host not driving IO0 in the instruction/address phase;
//    SCLK pulses with CS# low but the instruction phase missing the quad bit (n/a).
//  * Reads past the end of the image return 0xFF (erased flash) and are counted in over_read.
//  * SCLK edges while CS# is high are ignored (the 7-series STARTUPE2 startup edges).
`timescale 1ns/1ps
module qspi_flash_model #(
    parameter string IMG = "",
    parameter int SIZE = 1024,           // flash bytes modelled (BASE + image bytes)
    parameter int BASE = 0,              // image offset in the flash (the bitstream lives below it)
    parameter bit QUAD_INIT = 1'b1,      // CR1[1] (QUAD) power-up value
    parameter bit ALLOW_REG_CMDS = 1'b0, // also model RDID 9Fh, RDSR1 05h, RDCR 35h, WREN 06h, WRR 01h (single-bit)
    parameter int ADDR_BYTES = 3,
    parameter int DUMMY = 8,
    parameter real TV = 7.65,            // ns, falling SCLK to output valid
    parameter real THO = 2.0,            // ns, output hold after falling SCLK
    parameter real TCS = 10.0,           // ns, CS# high time between read commands
    parameter real TCSS = 0.0,           // ns, CS# active setup (CS# fall to first SCLK rise); 0 disables the check
    parameter real TCSH = 0.0,           // ns, CS# active hold (last SCLK rise to CS# rise); 0 disables
    parameter real TSU = 0.0,            // ns, host data in setup before SCLK rise; 0 disables
    parameter real THD = 0.0             // ns, host data in hold after SCLK rise; 0 disables
) (
    input  logic       sck, cs_n,
    input  logic [3:0] host_o, host_oe,
    output logic [3:0] flash_o, flash_oe
);
    localparam logic [7:0] OPCODE = (ADDR_BYTES == 4) ? 8'h6C : 8'h6B;
    logic [7:0] mem [SIZE];
    initial begin
        for (int i = 0; i < SIZE; i++) mem[i] = 8'hFF;
        $readmemh(IMG, mem, BASE);
        flash_o = 4'bz; flash_oe = 4'h0;
    end
    // phase: 0 idle, 1 instruction, 2 address, 3 dummy, 4 data
    int phase = 0, bitn = 0, rises = 0;          // rises: rising edges in this command
    logic [7:0] instr; logic [31:0] addr;
    int unsigned rd_addr; int nib = 0; int over_read = 0, nbytes_out = 0, ncmds = 0;
    realtime last_cs_rise = -1000.0, t_rise = -1000.0, t_chg = -1000.0, t_csfall = -1000.0;
    // host-to-flash setup/hold checks (instruction, address, WRR data on IO0)
    always @(host_o[0] or host_oe[0]) begin
        if (cs_n === 1'b0 && (phase == 1 || phase == 2 || phase == 7) && THD > 0.0 && $realtime - t_rise < THD && $realtime - t_rise >= 0.0 && t_rise > 0.0)
            $fatal(1, "FLASH: host data changed %0.2f ns after SCLK rise (tHD %0.1f)", $realtime - t_rise, THD);
        t_chg = $realtime;
    end
    logic [7:0] regsnap, sr1 = 8'h00, cr1 = {6'b0, QUAD_INIT, 1'b0}, regbyte, wrr_b0, wrr_b1;
    logic wel = 0; int wip_polls = 0, rbit = 0, ridx = 0; logic [7:0] id_bytes [6]; initial begin id_bytes[0] = 8'h01; id_bytes[1] = 8'h20; id_bytes[2] = 8'h18; id_bytes[3] = 8'h4D; id_bytes[4] = 8'h00; id_bytes[5] = 8'h80; end  // ID-CFI bytes 0..5, uniform 256 kB sectors
    int n_wrr = 0, n_quad_reads_blocked = 0;
    wire driving = (phase == 4);

    logic cs_low = 1'b0;     // x->0 / 1->x transitions at start-up are not real CS# edges
    always @(negedge cs_n) if (cs_n === 1'b0 && !cs_low) begin
        cs_low = 1'b1;
        if (sck !== 1'b0) $fatal(1, "FLASH: CS# fell while SCLK=%b (mode 0 requires SCLK low)", sck);
        if ($realtime - last_cs_rise < TCS) $fatal(1, "FLASH: CS# high time %0.2f ns < tCS %0.1f ns", $realtime - last_cs_rise, TCS);
        phase = 1; bitn = 0; rises = 0; instr = 0; addr = 0; nib = 0; t_csfall = $realtime; t_rise = -1000.0;
    end
    always @(posedge cs_n) if (cs_n === 1'b1 && cs_low) begin
        cs_low = 1'b0;
        if (sck !== 1'b0) $fatal(1, "FLASH: CS# rose while SCLK=%b", sck);
        if (TCSH > 0.0 && rises > 0 && $realtime - t_rise < TCSH) $fatal(1, "FLASH: CS# rose %0.2f ns after the last SCLK rise (tCSH %0.1f)", $realtime - t_rise, TCSH);
        if ((phase == 1 || phase == 2 || phase == 3) && rises % 8 != 0) $fatal(1, "FLASH: CS# raised mid-byte of the command (%0d rising edges)", rises);
        if (phase == 6) begin
            if (rises != 8) $fatal(1, "FLASH: WREN with %0d clocks", rises);
            wel = 1;
        end
        if (phase == 8) begin
            if (rises != 8) $fatal(1, "FLASH: WRDI with %0d clocks", rises);
            wel = 0;
        end
        if (phase == 7) begin
            if (rises != 24) $fatal(1, "FLASH: WRR must clock 24 bits (SR1+CR1), got %0d", rises);
            if (wel) begin
                cr1 = wrr_b1; sr1 = {wrr_b0[7:2], 2'b00}; wel = 0; wip_polls = 2; n_wrr++;
            end
        end
        phase = 0; last_cs_rise = $realtime;
        flash_o = 4'bz; flash_oe = 4'h0;
    end
    always @(posedge sck) if (cs_n === 1'b0) begin
        rises++;
        if (rises == 1 && TCSS > 0.0 && $realtime - t_csfall < TCSS) $fatal(1, "FLASH: CS# fell %0.2f ns before the first SCLK rise (tCSS %0.1f)", $realtime - t_csfall, TCSS);
        if (TSU > 0.0 && (phase == 1 || phase == 2 || phase == 7) && $realtime - t_chg < TSU)
            $fatal(1, "FLASH: host data changed %0.2f ns before SCLK rise (tSU %0.1f)", $realtime - t_chg, TSU);
        t_rise = $realtime;
        case (phase)
            1: begin
                if (host_oe[0] !== 1'b1) $fatal(1, "FLASH: host not driving IO0 during instruction");
                instr = {instr[6:0], host_o[0]}; bitn++;
                if (bitn == 8) begin
                    bitn = 0;
                    if (instr === OPCODE) begin
                        if (!cr1[1]) $fatal(1, "FLASH: Quad Output Read issued with CR1.QUAD = 0");
                        phase = 2;
                    end else if (ALLOW_REG_CMDS && (instr === 8'h9F || instr === 8'h05 || instr === 8'h35)) begin
                        phase = 5; ridx = 0; rbit = 0;
                        regsnap = {sr1[7:2], wel, (wip_polls > 0)}; if (instr === 8'h05 && wip_polls > 0) wip_polls--;
                    end else if (ALLOW_REG_CMDS && instr === 8'h06) phase = 6;
                    else if (ALLOW_REG_CMDS && instr === 8'h04) phase = 8;     // WRDI
                    else if (ALLOW_REG_CMDS && instr === 8'h01) phase = 7;
                    else $fatal(1, "FLASH: instruction %02h, expected %02h", instr, OPCODE);
                end
            end
            2: begin
                if (host_oe[0] !== 1'b1) $fatal(1, "FLASH: host not driving IO0 during address");
                addr = {addr[30:0], host_o[0]}; bitn++;
                if (bitn == 8 * ADDR_BYTES) begin
                    if (addr >= SIZE) $fatal(1, "FLASH: address %0d beyond the image (%0d bytes)", addr, SIZE);
                    rd_addr = addr; phase = 3; bitn = 0; ncmds++;
                    if (DUMMY == 0) phase = 4;
                end
            end
            3: begin bitn++; if (bitn == DUMMY) phase = 4; end   // data launches on the next falling edge
            7: begin
                if (host_oe[0] !== 1'b1) $fatal(1, "FLASH: host not driving IO0 during WRR data");
                if (bitn < 8) wrr_b0 = {wrr_b0[6:0], host_o[0]}; else wrr_b1 = {wrr_b1[6:0], host_o[0]};
                bitn++;
            end
            4: begin
                if (host_oe !== 4'h0) $fatal(1, "FLASH: host drives the bus (oe=%b) while the flash outputs data", host_oe);
            end
        endcase
    end
    // single-bit register reads: SO = IO1, launched on the falling edge
    always @(negedge sck) if (cs_n === 1'b0 && phase == 5) begin
        if (host_oe[1] !== 1'b0) $fatal(1, "FLASH: host drives IO1 (SO) during a register read");
        if (rbit == 0) begin
            case (instr)
                8'h9F: regbyte = id_bytes[ridx % 6];
                8'h05: regbyte = regsnap;
                default: regbyte = cr1;
            endcase
        end
        #(THO) if (cs_n === 1'b0) begin flash_o = 4'bx; flash_oe = 4'b0010; end
        #(TV - THO) if (cs_n === 1'b0) flash_o = {2'b00, regbyte[7 - rbit], 1'b0};
        rbit++; if (rbit == 8) begin rbit = 0; ridx++; end
    end
    // data launch: after the last dummy rising edge, each falling edge starts the next nibble
    always @(negedge sck) if (cs_n === 1'b0 && phase == 4) begin
        logic [7:0] b; logic [3:0] v;
        b = (rd_addr < SIZE) ? mem[rd_addr] : 8'hFF;
        v = (nib == 0) ? b[7:4] : b[3:0];
        if (rd_addr >= SIZE && nib == 0) over_read++;
        if (host_oe !== 4'h0) $fatal(1, "FLASH: host drives the bus (oe=%b) at a data launch", host_oe);
        #(THO) if (cs_n === 1'b0) begin flash_o = 4'bx; flash_oe = 4'hF; end
        #(TV - THO) if (cs_n === 1'b0) flash_o = v;
        if (nib == 1) begin rd_addr = (ADDR_BYTES == 4) ? rd_addr + 1 : (rd_addr + 1) & 32'hFFFFFF; nbytes_out++; end
        nib = 1 - nib;
    end
endmodule
