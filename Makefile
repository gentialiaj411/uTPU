
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

XVLOG ?= xvlog
XELAB ?= xelab
XSIM  ?= xsim

# Vivado-free RTL sim tools (Phase 0 harness). Override if installed elsewhere.
IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python

SIM_OUT          ?= build/sim
SIM_IVERILOG_OUT ?= build/sim_iverilog
VIVADO_SETTINGS  ?= /opt/Xilinx/Vivado/2024.1/settings64.sh

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

# ---------------------------
# Vivado-free RTL sim (icarus). Reuses existing rtl/tb/* testbenches.
# Emits VCD into $(SIM_IVERILOG_OUT) for debugging without Vivado.
# Default behavior of the testbenches is unchanged when DUMP_VCD is absent.
# ---------------------------
RTL_DESIGN := \
	rtl/top/top.sv \
	rtl/memory/instr_bram.sv \
	rtl/PEArray/pe_controller.sv \
	rtl/PEArray/pe_array.sv \
	rtl/PEArray/pe.sv \
	rtl/quantizer/quantizer.sv \
	rtl/quantizer/quantizer_array.sv \
	rtl/LeakyReLU/leaky_relu.sv \
	rtl/LeakyReLU/leaky_relu_array.sv \
	rtl/unified_buffer/unified_buffer.sv \
	rtl/fifo/fifo_rx.sv \
	rtl/fifo/fifo_tx.sv \
	rtl/UART/uart.sv \
	rtl/UART/uart_receiver.sv \
	rtl/UART/uart_transmitter.sv \
	rtl/UART/clk_divider.sv

RTL_STUBS := rtl/tb/xpm_memory_sdpram_stub.sv

.PHONY: sim-units sim-all sim-perf clean-sim bench test-host \
        sim-iverilog-fused sim-iverilog-perf sim-iverilog-all clean-sim-iverilog \
        repro repro-host repro-cuda

# ---------------------------
# HOST REGRESSION (single source of truth for the CI test list)
# Keep this list byte-identical to the "Run tests" step in
# .github/workflows/ci.yml so docs/agents never drift.
# ---------------------------
test-host:
	python -m pytest firmware/host/test_fx_importer.py firmware/host/test_pytorch_compiler.py firmware/host/test_torch_compile_backend.py firmware/host/test_compiled_runtime_validation.py firmware/host/test_footprint_baseline.py firmware/host/test_compiler_smoke.py firmware/host/test_graph_passes.py firmware/host/test_reference_interpreter.py firmware/host/test_isa_simulator.py firmware/host/test_rtl_sim_artifact.py firmware/host/test_transformer_integration.py firmware/host/test_cost_model_regression.py firmware/host/test_cost_model_selection.py firmware/host/test_fusion_benchmark.py firmware/host/test_multi_pe_sim.py firmware/host/test_real_model_ops.py firmware/host/test_real_model_end_to_end.py -v

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

sim-perf:
	mkdir -p "$(SIM_OUT)"
	source "$(VIVADO_SETTINGS)"
	cd "$(SIM_OUT)"
	"$(XVLOG)" -sv $(INCDIRS) $(RTL_ALL) $(CURDIR)/rtl/tb/tb_perf_counters.sv $(CURDIR)/rtl/tb/xpm_memory_sdpram_stub.sv |& tee compile-perf.log
	"$(XELAB)" "tb_perf_counters" -debug typical |& tee elab-perf.log
	"$(XSIM)"  "tb_perf_counters" -runall |& tee run-perf.log

clean-sim:
	rm -rf "$(SIM_OUT)"

bench:
	# Writes benchmarks/summary.json via firmware/host/lock_benchmarks.py.
	python firmware/host/lock_benchmarks.py --runs 5

# ---------------------------
# Vivado-free RTL sim targets (Phase 0). Require icarus verilog (iverilog/vvp).
# Linux/WSL2:   sudo apt-get install -y iverilog
# Windows:      install from https://bleyer.org/icarus/ (default path C:\iverilog\bin)
# These targets are local-only (not in CI). Status: verify before relying.
# ---------------------------
sim-iverilog-fused:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/test_vectors build/reports
	$(PYTHON) firmware/host/generate_fused_rtl_test_vectors.py > /dev/null
	"$(IVERILOG)" -g2012 -DICARUS -DDUMP_VCD \
		-o "$(SIM_IVERILOG_OUT)/tb_fused_compressed_program.out" \
		rtl/tb/tb_fused_compressed_program.sv $(RTL_STUBS) $(RTL_DESIGN)
	"$(VVP)" "$(SIM_IVERILOG_OUT)/tb_fused_compressed_program.out"
	@echo "[sim-iverilog-fused] VCD: $(SIM_IVERILOG_OUT)/tb_fused_compressed_program.vcd"

sim-iverilog-perf:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/reports
	"$(IVERILOG)" -g2012 -DICARUS -DDUMP_VCD \
		-o "$(SIM_IVERILOG_OUT)/tb_perf_counters.out" \
		rtl/tb/tb_perf_counters.sv $(RTL_STUBS) $(RTL_DESIGN)
	"$(VVP)" "$(SIM_IVERILOG_OUT)/tb_perf_counters.out"
	@echo "[sim-iverilog-perf] VCD: $(SIM_IVERILOG_OUT)/tb_perf_counters.vcd"

sim-iverilog-all: sim-iverilog-fused sim-iverilog-perf

clean-sim-iverilog:
	rm -rf "$(SIM_IVERILOG_OUT)"

# ---------------------------
# One-command repro of every existing headline artifact.
# `repro-host`  : artifacts that do not require CUDA (runs on Windows/Linux/macOS).
# `repro-cuda`  : ResNet-18 end-to-end + parity tests (require Linux/WSL2 + CUDA + gcc).
# `repro`       : repro-host; for CUDA artifacts run `make repro-cuda` separately.
# No new metrics are produced here - only existing scripts are re-invoked.
# ---------------------------
repro-host:
	mkdir -p bench/results build/reports
	@echo "[repro-host] cost-model regression -> bench/results/cost_model_regression.json"
	$(PYTHON) firmware/host/run_cost_model_regression.py
	@echo "[repro-host] multi-PE ISA sim -> bench/results/multi_pe_sim.json"
	$(PYTHON) firmware/host/run_multi_pe_sim_benchmark.py
	@echo "[repro-host] ISA <-> RTL bitmatch (iverilog) -> build/reports/isa_rtl_bitmatch_report.json"
	$(PYTHON) firmware/host/run_isa_rtl_bitmatch.py \
		--output-json build/reports/isa_rtl_bitmatch_report.json \
		--output-md   build/reports/isa_rtl_bitmatch_report.md
	@echo "[repro-host] differential harness -> build/reports/differential_test_report.json"
	$(PYTHON) -m pytest firmware/host/test_differential_harness.py -q
	@echo "[repro-host] pruned autotuner replay -> build/reports/pruned_autotuner_report.json"
	$(PYTHON) firmware/host/evaluate_pruned_autotuner.py --top-k 4 \
		--output-json build/reports/pruned_autotuner_report.json \
		--output-md   build/reports/pruned_autotuner_report.md
	@echo "[repro-host] cost-model selection (Phase 1) -> bench/results/cost_model_selection.json"
	$(PYTHON) firmware/host/run_cost_model_selection.py \
		--output-json bench/results/cost_model_selection.json
	@echo "[repro-host] fusion payoff (Phase 2) -> bench/results/fusion_payoff.json"
	$(PYTHON) firmware/host/run_fusion_benchmark.py
	@echo "[repro-host] done. CUDA-only artifact (ResNet-18) is regenerated via 'make repro-cuda'."

repro-cuda:
	mkdir -p bench/results
	@echo "[repro-cuda] ResNet-18 end-to-end (requires CUDA + Linux/WSL2) -> bench/results/real_model_end_to_end.json"
	$(PYTHON) firmware/host/run_real_model_end_to_end.py --output bench/results/real_model_end_to_end.json
	$(PYTHON) -m pytest firmware/host/test_real_model_ops.py firmware/host/test_real_model_end_to_end.py -q

repro: repro-host
	@echo "[repro] host artifacts regenerated. For ResNet-18 (CUDA), run: make repro-cuda"
