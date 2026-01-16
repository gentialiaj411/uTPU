
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

XVLOG ?= xvlog
XELAB ?= xelab
XSIM  ?= xsim

SIM_OUT ?= build/sim

# Set this to your actual Vivado settings script location
VIVADO_SETTINGS ?= /opt/Xilinx/Vivado/2024.1/settings64.sh

# Absolute include dirs (so `include works after cd)
INCDIRS := -i $(CURDIR)/rtl/PEArray -i $(CURDIR)/generated

# Absolute sources
PEARRAY_RTL := $(CURDIR)/rtl/PEArray/pe.sv $(CURDIR)/rtl/PEArray/pe_array.sv
PEARRAY_TB  := $(CURDIR)/sim/PEArray/pe_array_tb2.sv


# Controller TB sources (absolute paths)
CTRL_RTL := $(CURDIR)/rtl/PEArray/pe_controller.sv
CTRL_TB  := $(CURDIR)/sim/PEArray/pe_controller_tb.sv

TOP_CTRL ?= pe_controller_tb
TOP ?= pe_array_tb2

.PHONY: sim-controller sim-pearray clean-sim

sim-controller:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(CTRL_RTL) $(CTRL_TB)
	"$(XELAB)" "$(TOP_CTRL)" -debug typical
	"$(XSIM)"  "$(TOP_CTRL)" -runall

sim-pearray:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(PEARRAY_RTL) $(PEARRAY_TB)
	"$(XELAB)" "$(TOP)" -debug typical
	"$(XSIM)" "$(TOP)" -runall

clean-sim:
	rm -rf "$(SIM_OUT)"

