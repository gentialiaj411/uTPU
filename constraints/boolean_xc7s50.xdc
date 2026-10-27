# Real Digital Boolean (xc7s50csga324-1) constraints for rtl/board/boolean_top.sv.
# Every pin below is checked by tools/board/pin_check.py against the device database (Vivado PIN_FUNC) and the
# Boolean schematic; results in bench/results/boolean_pin_check.json. clk, UART_rxd and UART_txd are the names and
# pins of Real Digital's master constraints file. The flash pins are not in that file; they come from schematic
# sheet 11 and are the dedicated configuration pins of bank 14 used as user I/O after configuration.
# The flash clock (CCLK_0, pin C8) is driven through STARTUPE2 (rtl/board/qspi_pads.sv) and needs no constraint.

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
# The flash data/CS timing relative to the STARTUPE2 clock is not constrained here (see docs/r3_progress.md, M5 open issues).
