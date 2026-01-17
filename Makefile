
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

XVLOG ?= xvlog
XELAB ?= xelab
XSIM  ?= xsim

SIM_OUT ?= build/sim
VIVADO_SETTINGS ?= /opt/Xilinx/Vivado/2024.1/settings64.sh

INCDIRS := \
	-i $(CURDIR)/rtl/PEArray \
	-i $(CURDIR)/rtl/fifo \
	-i $(CURDIR)/rtl/LeakyReLU \
	-i $(CURDIR)/rtl/quantizer \
	-i $(CURDIR)/rtl/top \
	-i $(CURDIR)/rtl/UART \
	-i $(CURDIR)/rtl/unified_buffer \
	-i $(CURDIR)/generated

# ---------------------------
# UNIT TESTS (no top.sv)
# ---------------------------
UNIT_RTL := \
  $(CURDIR)/rtl/fifo/fifo.sv \
  $(CURDIR)/rtl/fifo/fifo_rx.sv \
  $(CURDIR)/rtl/fifo/fifo_tx.sv \
  $(CURDIR)/rtl/quantizer/quantizer.sv \
  $(CURDIR)/rtl/quantizer/quantizer_array.sv \
  $(CURDIR)/rtl/LeakyReLU/leaky_relu.sv \
  $(CURDIR)/rtl/LeakyReLU/leaky_relu_array.sv

UNIT_TB  := $(CURDIR)/sim/top/units_tb.sv
TOP_UNITS ?= units_tb

# ---------------------------
# FULL INTEGRATION (everything + top)
# ---------------------------
RTL_ALL := \
	$(CURDIR)/rtl/fifo/fifo.sv \
	$(CURDIR)/rtl/fifo/fifo_rx.sv \
	$(CURDIR)/rtl/fifo/fifo_tx.sv \
	$(CURDIR)/rtl/LeakyReLU/leaky_relu.sv \
	$(CURDIR)/rtl/LeakyReLU/leaky_relu_array.sv \
	$(CURDIR)/rtl/quantizer/quantizer.sv \
	$(CURDIR)/rtl/quantizer/quantizer_array.sv \
	$(CURDIR)/rtl/PEArray/pe.sv \
	$(CURDIR)/rtl/PEArray/pe_array.sv \
	$(CURDIR)/rtl/PEArray/pe_controller.sv \
	$(CURDIR)/rtl/unified_buffer/unified_buffer.sv \
	$(CURDIR)/rtl/UART/clk_divider.sv \
	$(CURDIR)/rtl/UART/uart_receiver.sv \
	$(CURDIR)/rtl/UART/uart_transmitter.sv \
	$(CURDIR)/rtl/UART/uart.sv \
	$(CURDIR)/rtl/top/top.sv

ALL_TB   := $(CURDIR)/sim/top/system_tb.sv
TOP_ALL  ?= system_tb

.PHONY: sim-units sim-all clean-sim

sim-units:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(UNIT_RTL) $(UNIT_TB) |& tee compile-units.log
	"$(XELAB)" "$(TOP_UNITS)" -debug typical            |& tee elab-units.log
	"$(XSIM)"  "$(TOP_UNITS)" -runall                  |& tee run-units.log

sim-all:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(RTL_ALL) $(ALL_TB)     |& tee compile-all.log
	"$(XELAB)" "$(TOP_ALL)" -debug typical             |& tee elab-all.log
	"$(XSIM)"  "$(TOP_ALL)" -runall                    |& tee run-all.log

clean-sim:
	rm -rf "$(SIM_OUT)"
