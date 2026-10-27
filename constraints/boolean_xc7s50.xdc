# Real Digital Boolean (xc7s50csga324-1) constraints for rtl/board/boolean_top.sv.
# Every pin below is checked by tools/board/pin_check.py against the device database (Vivado PIN_FUNC) and the
# Boolean schematic; results in bench/results/boolean_pin_check.json. clk, UART_rxd and UART_txd are the names and
# pins of Real Digital's master constraints file. The flash pins are not in that file; they come from schematic
# sheet 11 and are the dedicated configuration pins of bank 14 used as user I/O after configuration.
# The flash clock (CCLK_0, pin C8) is driven through STARTUPE2 (rtl/board/qspi_pads.sv) and has no port; it is timed with a
# generated clock on USRCCLKO below.

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# 100 MHz oscillator (IC9), bank 15
set_property -dict {PACKAGE_PIN F14 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# USB-UART bridge, bank 14 (names are from the FPGA's point of view, as in the master XDC)
set_property -dict {PACKAGE_PIN V12 IOSTANDARD LVCMOS33} [get_ports UART_rxd]
set_property -dict {PACKAGE_PIN U11 IOSTANDARD LVCMOS33} [get_ports UART_txd]
set_false_path -from [get_ports UART_rxd]
set_false_path -to [get_ports UART_txd]

# S25FL128S quad SPI flash, bank 14 (VCCO 3.3 V)
set_property -dict {PACKAGE_PIN M13 IOSTANDARD LVCMOS33} [get_ports qspi_cs_n]
set_property -dict {PACKAGE_PIN K17 IOSTANDARD LVCMOS33} [get_ports {qspi_dq[0]}]
set_property -dict {PACKAGE_PIN K18 IOSTANDARD LVCMOS33} [get_ports {qspi_dq[1]}]
set_property -dict {PACKAGE_PIN L14 IOSTANDARD LVCMOS33} [get_ports {qspi_dq[2]}]
set_property -dict {PACKAGE_PIN M15 IOSTANDARD LVCMOS33} [get_ports {qspi_dq[3]}]

# ---------------------------------------------------------------------------------------------------------------------
# Flash interface timing. Valid for the shipped configuration only: qspi_reader HALF=1 (SCLK = clk50/2 = 25 MHz), SAMPLE_LATE=1,
# CAP_DLY=2, all flash-facing registers in the IOBs (qspi_pad_regs). Numbers and sources:
#   S25FL128S datasheet rev 05 (Spansion, 12 Jul 2012), table 6.4 "AC Characteristics", VIO = VCC 2.7-3.6 V:
#       tSU  data in setup          min 3 ns        tHD  data in hold        min 2 ns
#       tCSS CS# active setup       min 3 ns        tCSH CS# active hold     min 3 ns
#       tV   clock low to output valid  max 8.0 ns (full VCC range, CL = 30 pF; 7.65 ns for 3.0-3.6 V regulated, 6.5 ns at 15 pF)
#       tHO  output hold            min 2 ns
#   DS189 Spartan-7 data sheet v1.11, Configuration Switching Characteristics, speed grade -1 (as read from the AMD docs page):
#       TUSRCCLKO  STARTUPE2 USRCCLKO input to CCLK pin: 0.50 ns min, 7.50 ns max
#   NOT from an official source: board trace delay (the Boolean layout is not public). Bounded assumption 0.0 .. 0.5 ns on
#   each of clock, data and CS# (about 3 inches); the worst case is constrained.
# ---------------------------------------------------------------------------------------------------------------------
set tsu_flash  3.0
set thd_flash  2.0
set tcss_flash 3.0
set tcsh_flash 3.0
set tv_max     8.0
set tho_min    2.0
set cclk_min   0.5
set cclk_max   7.5
set trace_min  0.0
set trace_max  0.5

# SCLK as seen at the flash: the clock leaving the qspi_pad_regs SCLK register (a toggle clock, clk50/2 while it runs),
# delayed by the USRCCLKO -> CCLK pin delay (early/late = TUSRCCLKO min/max).
create_generated_clock -name clk_sck -source [get_pins -hierarchical -filter {NAME =~ *u_pr/sclk_r_reg/C}] -divide_by 2 [get_pins u_pads/u_startup/USRCCLKO]
set_clock_latency -source -early $cclk_min [get_clocks clk_sck]
set_clock_latency -source -late  $cclk_max [get_clocks clk_sck]

# Outputs (core -> flash): the flash samples on the SCLK rising edge. Setup = tSU / tCSS (+ trace), hold = -(tHD / tCSH) (+ min trace).
set_output_delay -clock clk_sck -max [expr {$tsu_flash + $trace_max}]  [get_ports {qspi_dq[*]}]
set_output_delay -clock clk_sck -min [expr {-$thd_flash + $trace_min}] [get_ports {qspi_dq[*]}]
set_output_delay -clock clk_sck -max [expr {$tcss_flash + $trace_max}] [get_ports qspi_cs_n]
set_output_delay -clock clk_sck -min [expr {-$tcsh_flash + $trace_min}] [get_ports qspi_cs_n]
# The output registers only change on the SCLK-falling phase (or while SCLK is parked low), never on the clk50 edge that
# coincides with an SCLK rising edge, so data launched at an edge is first sampled one cycle later and the same-edge hold
# relation the tool would otherwise check (launch and sample on the same clk50/clk_sck edge) does not occur. Move the hold
# check back one SCLK period; launch phases are guaranteed by rtl/decode/qspi_reader.sv (all io_o/io_oe/cs_n updates are
# on fall ticks, load/abort cycles with SCLK low) and exercised by the flash model's tSU/tHD/tCSS/tCSH checks.
set_multicycle_path 1 -hold -end -from [get_clocks clk50_raw] -to [get_ports {qspi_cs_n qspi_dq[*]}]

# Inputs (flash -> core): data is launched by the SCLK falling edge and is valid tV(max) + trace after it and held tHO(min) after
# the next falling edge. The input register samples it two core cycles (one SCLK period) later: setup multicycle 2, hold 1.
set_input_delay -clock clk_sck -clock_fall -max [expr {$tv_max + $trace_max}]  [get_ports {qspi_dq[*]}]
set_input_delay -clock clk_sck -clock_fall -min [expr {$tho_min + $trace_min}] [get_ports {qspi_dq[*]}]
set_multicycle_path 2 -setup -end -from [get_clocks clk_sck] -to [get_clocks clk50_raw]
set_multicycle_path 1 -hold  -end -from [get_clocks clk_sck] -to [get_clocks clk50_raw]
