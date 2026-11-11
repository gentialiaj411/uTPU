# DDR flash-read overlay for rtl/board/boolean_top.sv built with DDR=1 (qspi_reader EDh, DDR Quad I/O Read).
# READ THIS FILE ONLY FOR A DDR=1 BUILD, AND AFTER constraints/boolean_xc7s50.xdc (read_xdc boolean_xc7s50.xdc; read_xdc this
# file). A DDR=0 build must not read it: its constraints are boolean_xc7s50.xdc alone, unchanged. Plain XDC has no `if`, so the
# DDR timing is a separate file instead of a conditional block. It re-states the SDR input exceptions of the base file (later
# constraints on the same objects win) and adds the falling-edge delays; pins, clocks and every other constraint come from
# the base file. Status: written for review, NOT yet run through Vivado (no implementation of the DDR build has been done).
#
# Valid only for: qspi_reader DDR=1, HALF=1 (SCK = clk50/2 = 25 MHz in the dummy/data phases, 12.5 MHz in the address/mode
# phase), CAP_DLY=2, all flash-facing registers in the IOBs (qspi_pad_regs), as for the base file.
#
# Sources. S25FL128S/S25FL256S datasheet 001-98283 Rev. *T (Infineon, 2025-04-21):
#   Table 15 "AC characteristics -- DDR operation" (p. 44), 66 MHz column (the 25 MHz SCK used here is below it):
#       tV   clock to output valid    max 6.5 ns, note 37: regulated VCC 3.0-3.6 V and CL = 15 pF (the only DDR value given)
#       tHO  output hold              min 1.5 ns
#       tSU / tHD IO in setup / hold  min 2.0 / 2.0 ns
#       Fig. 41 "SPI DDR output timing" (p. 45): tV and tHO apply to both SCK edges.
#   Table 13 "AC characteristics" SDR (p. 40): tV max 8.0 ns at full VCC 2.7-3.6 V and CL = 30 pF (7.65 ns regulated 30 pF,
#       6.5 ns regulated 15 pF, i.e. the same 6.5 ns as Table 15 at the same conditions). The board's flash load is not known,
#       so the 30 pF / full-range SDR value 8.0 ns is used for DDR as well (conservative; the datasheet gives no DDR value
#       for 30 pF). With 6.5 ns the input setup margin below grows by 1.5 ns.
#   TUSRCCLKO 0.50 .. 7.50 ns (STARTUPE2 USRCCLKO -> CCLK pin, DS189 -1) and the 0 .. 0.5 ns board trace assumption: as in
#   the base file (same clock definition clk_sck, same source latency), not repeated here.
#
# Inputs (flash -> core). In the data phase every SCK toggle launches one nibble (falling: high nibble, rising: low nibble;
# Fig. 94 p. 105), and the IOB input register samples it ONE core cycle (20 ns) after the core edge that toggled the SCLK
# register, i.e. at the core edge of the next toggle (qspi_reader header). Relative to that ideal USRCCLKO edge E:
#   data valid at the FPGA pin   E + 7.5 (TUSRCCLKO max) + 8.0 (tV) + 0.5 (trace max)          = E + 16.0 ns
#   earliest change (next edge)  E + 20 + 0.5 (TUSRCCLKO min) + 1.5 (tHO) + 0.0 (trace min)    = E + 22.0 ns
#   sample                       E + 20 ns: ideal setup margin 4.0 ns, ideal hold margin 2.0 ns (before IBUF/IOB setup/hold
#   and clock insertion; for the SDR build Vivado reported +21.5 setup / +2.95 hold against ideal 24.0 / 2.5 on the same
#   pads, which suggests about +1.5 ns setup and +2.4 ns hold here: an ESTIMATE, not a Vivado result).
# That is exactly Vivado's default single-cycle relation from either clk_sck edge to the next clk50 edge (setup) and to the
# same clk50 edge (hold: data launched by the clk_sck edge at E must not reach the register before the clk50 edge at E has
# sampled the previous nibble). The base file's multicycle 2 (setup) / 1 (hold) from clk_sck to clk50 describes the SDR sample
# one SCLK period after the launch and must NOT apply here, so it is set back to the default below.
set tv_ddr_max   8.0
set tho_ddr_min  1.5
set tsu_ddr      2.0
set tsu_out      3.0   ;# output setup used for both SCK edges: max(SDR tSU 3.0 of the base file, DDR tSU 2.0)
set thd_ddr      2.0
set cclk_trace_max 0.5
set cclk_trace_min 0.0

set_input_delay -clock clk_sck -clock_fall -max [expr {$tv_ddr_max + $cclk_trace_max}]  [get_ports {qspi_dq[*]}]
set_input_delay -clock clk_sck -clock_fall -min [expr {$tho_ddr_min + $cclk_trace_min}] [get_ports {qspi_dq[*]}]
set_input_delay -clock clk_sck -max [expr {$tv_ddr_max + $cclk_trace_max}]  -add_delay [get_ports {qspi_dq[*]}]
set_input_delay -clock clk_sck -min [expr {$tho_ddr_min + $cclk_trace_min}] -add_delay [get_ports {qspi_dq[*]}]
set_multicycle_path 1 -setup -end -from [get_clocks clk_sck] -to [get_clocks -of_objects [get_pins u_mmcm/CLKOUT0]]
set_multicycle_path 0 -hold  -end -from [get_clocks clk_sck] -to [get_clocks -of_objects [get_pins u_mmcm/CLKOUT0]]

# Outputs (core -> flash). The instruction (SDR, rising-edge sampled) is timed by the base file. Address and mode nibbles are
# sampled on BOTH SCK edges (Fig. 94, Fig. 40 "SPI DDR input timing" p. 44), so falling-edge output delays are added for
# IO[3:0] (IO1 has an output path in the DDR build). The SDR tSU 3.0 ns (base file) is larger than the DDR 2.0 ns and is
# kept for both edges; tHD is 2.0 ns either way.
#   Setup (Vivado): a nibble launched on one clk50 edge must meet tSU at the next SCK edge, which Vivado assumes one core cycle
#   later (worst case of the generated-clock model); the RTL launches each address/mode nibble exactly one core cycle before
#   its sampling edge, so this check is exact: 20 ns - (TUSRCCLKO min 0.5 vs output path) - tSU 3.0 - trace 0.5.
#   Hold: the RTL changes an address/mode nibble only on a core edge WITHOUT an SCK edge, one core cycle after the edge that
#   sampled the previous nibble. Hand margin: 20 - 7.5 (TUSRCCLKO max) - 0.5 (trace) - 2.0 (tHD) = +10 ns plus the IOB
#   clock-to-out. Vivado's check follows the base file's multicycle -hold 1 -end on clk50 -> qspi ports (one SCLK period),
#   which for these falling-edge delays, as for the base file's rising-edge ones, is more relaxed than the real 20 ns
#   relation; the real hold margin is the hand value above (kept by RTL construction, exercised by the flash model's
#   tSU/tHD checks in simulation).
#   Note on the base file's multicycle comment ("output registers never change on the clk50 edge of an SCLK rising edge"): in
#   the DDR build the IO output enables are released on the core edge of the FIRST DUMMY rising edge (bus turn-around). That
#   edge samples nothing (dummy cycle), the last mode nibble was sampled on the falling edge one core cycle earlier, and the
#   flash does not drive before the falling edge that ends the last dummy cycle (DLP off), so no sampled value depends on it.
set_output_delay -clock clk_sck -clock_fall -max [expr {$tsu_out + $cclk_trace_max}]  -add_delay [get_ports {qspi_dq[*]}]
set_output_delay -clock clk_sck -clock_fall -min [expr {-$thd_ddr + $cclk_trace_min}] -add_delay [get_ports {qspi_dq[*]}]
