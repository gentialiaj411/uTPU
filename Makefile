
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

XVLOG ?= xvlog
XELAB ?= xelab
XSIM  ?= xsim

SIM_OUT ?= build/sim

VIVADO_SETTINGS ?= /opt/Xilinx/Vivado/2024.1/settings64.sh

# ------------------------------------------------------------
# Include dirs (absolute, so they still work after cd)
# ------------------------------------------------------------
INCDIRS := \
	-i $(CURDIR)/rtl/PEArray \
	-i $(CURDIR)/rtl/fifo \
	-i $(CURDIR)/rtl/LeakyReLU \
	-i $(CURDIR)/rtl/quantizer \
	-i $(CURDIR)/rtl/top \
	-i $(CURDIR)/rtl/UART \
	-i $(CURDIR)/rtl/unified_buffer \
	-i $(CURDIR)/generated

# ------------------------------------------------------------
# Existing per-block sources (kept)
# ------------------------------------------------------------
PEARRAY_RTL := $(CURDIR)/rtl/PEArray/pe.sv $(CURDIR)/rtl/PEArray/pe_array.sv
PEARRAY_TB  := $(CURDIR)/sim/PEArray/pe_array_tb2.sv

CTRL_RTL := $(CURDIR)/rtl/PEArray/pe_controller.sv
CTRL_TB  := $(CURDIR)/sim/PEArray/pe_controller_tb.sv

TOP_CTRL ?= pe_controller_tb
TOP_PE   ?= pe_array_tb2

# ------------------------------------------------------------
# TBs
# ------------------------------------------------------------
UNIT_TB := $(CURDIR)/sim/top/top_tb.sv
TB_ALL  := $(CURDIR)/sim/top/top_tb.sv
TOP_ALL ?= top_tb

# ------------------------------------------------------------
# RTL lists
# ------------------------------------------------------------
UNIT_RTL := \
	$(CURDIR)/rtl/fifo/fifo.sv \
	$(CURDIR)/rtl/fifo/fifo_rx.sv \
	$(CURDIR)/rtl/fifo/fifo_tx.sv \
	$(CURDIR)/rtl/quantizer/quantizer.sv \
	$(CURDIR)/rtl/quantizer/quantizer_array.sv \
	$(CURDIR)/rtl/LeakyReLU/leaky_relu.sv \
	$(CURDIR)/rtl/LeakyReLU/leaky_relu_array.sv

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

.PHONY: sim-units sim-all sim-controller sim-pearray clean-sim show-errors-units show-errors-all

sim-units:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(UNIT_RTL) $(UNIT_TB) |& tee compile-units.log
	"$(XELAB)" top_tb -debug typical |& tee elab-units.log
	"$(XSIM)"  top_tb -runall     |& tee run-units.log

sim-all:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(RTL_ALL) $(TB_ALL) |& tee compile-all.log
	"$(XELAB)" "$(TOP_ALL)" -debug typical |& tee elab-all.log
	"$(XSIM)"  "$(TOP_ALL)" -runall        |& tee run-all.log

sim-controller:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(CTRL_RTL) $(CTRL_TB) |& tee compile-controller.log
	"$(XELAB)" "$(TOP_CTRL)" -debug typical |& tee elab-controller.log
	"$(XSIM)"  "$(TOP_CTRL)" -runall        |& tee run-controller.log

sim-pearray:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(PEARRAY_RTL) $(PEARRAY_TB) |& tee compile-pearray.log
	"$(XELAB)" "$(TOP_PE)" -debug typical |& tee elab-pearray.log
	"$(XSIM)"  "$(TOP_PE)" -runall       |& tee run-pearray.log

show-errors-units:
	@cd "$(SIM_OUT)" && \
	if [ -f compile-units.log ]; then \
	  grep -n "ERROR:" compile-units.log | head -n 50; \
	else \
	  echo "No compile-units.log found. Run: make sim-units"; \
	fi

show-errors-all:
	@cd "$(SIM_OUT)" && \
	if [ -f compile-all.log ]; then \
	  grep -n "ERROR:" compile-all.log | head -n 50; \
	else \
	  echo "No compile-all.log found. Run: make sim-all"; \
	fi

clean-sim:
	rm -rf "$(SIM_OUT)"
