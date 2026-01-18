## Arty A7 Rev E constraints for 100 MHz clock (E3) and USB-UART.
## Ports assumed: clk, rx, tx
## Reset pin
set_property -dict { PACKAGE_PIN C2 IOSTANDARD LVCMOS33 } [get_ports { rst }];
## Reset indicator LED
set_property -dict { PACKAGE_PIN H5 IOSTANDARD LVCMOS33 } [get_ports { led_rst }];

## Configuration bank voltage
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## 100 MHz onboard oscillator (user-provided clock pin)
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports { clk }]; # Sch=gclk[100]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports { clk }];

## USB-UART bridge (FTDI) pins
## rx = FPGA input from USB-UART (PC TX)
## tx = FPGA output to USB-UART (PC RX)
set_property -dict { PACKAGE_PIN A9  IOSTANDARD LVCMOS33 } [get_ports { rx }]; # uart_txd_in
set_property -dict { PACKAGE_PIN D10 IOSTANDARD LVCMOS33 } [get_ports { tx }]; # uart_rxd_out
