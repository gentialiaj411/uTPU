## Real Digital Boolean board (xc7s50csga324-1) uTPU top-level constraints.
## Sources: Boolean Reference Manual https://www.realdigital.org/doc/02013cd17602c8af749f00561f88ae21
## and master XDC https://www.realdigital.org/downloads/8d5c167add28c014173edcf51db78bb9.txt.
## Schematic https://www.realdigital.org/downloads/63f9a8205ebd9c2e8c2d265ad25097dc.pdf:
## p.3 shows active-high BTN0; p.9 shows PROG USB-UART (FPGA TX=U11, RX=V12).
## The official Boolean tutorial identifies the 100 MHz oscillator on F14,
## BTN0 on J2, and LED0 on G1.  The official schematic identifies the USB-UART
## nets UART_RXD_OUT=U11 (FPGA TX) and UART_TXD_IN=V12 (FPGA RX).
## The PROG UART micro-USB connection supplies both JTAG programming and UART.

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## 100 MHz oscillator
set_property -dict { PACKAGE_PIN F14 IOSTANDARD LVCMOS33 } [get_ports { clk }]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports { clk }]

## BTN0, active-high reset input
set_property -dict { PACKAGE_PIN J2 IOSTANDARD LVCMOS33 } [get_ports { rst }]

## USB-UART bridge: rx is PC->FPGA, tx is FPGA->PC
set_property -dict { PACKAGE_PIN V12 IOSTANDARD LVCMOS33 } [get_ports { rx }]
set_property -dict { PACKAGE_PIN U11 IOSTANDARD LVCMOS33 } [get_ports { tx }]

## LED0 is used by top.sv as the reset/status indicator.
set_property -dict { PACKAGE_PIN G1 IOSTANDARD LVCMOS33 } [get_ports { led_rst }]
