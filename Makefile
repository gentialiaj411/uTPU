
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
        sim-iverilog-fused sim-iverilog-perf sim-iverilog-batched sim-iverilog-phase4-widen \
        sim-iverilog-scheduler-cross-check sim-iverilog-latency \
        sim-iverilog-all clean-sim-iverilog \
        repro repro-host repro-cuda \
        fuzz fuzz-discovery superopt latency-determinism \
        megakernel-recompute-aggregate nsight-compute-profile \
        cublas-baseline-recompute-aggregate

# ---------------------------
# HOST REGRESSION (single source of truth for the CI test list)
# Keep this list byte-identical to the "Run tests" step in
# .github/workflows/ci.yml so docs/agents never drift.
# ---------------------------
test-host:
	python -m pytest firmware/host/test_fx_importer.py firmware/host/test_pytorch_compiler.py firmware/host/test_torch_compile_backend.py firmware/host/test_compiled_runtime_validation.py firmware/host/test_compiled_runtime_schedule_source.py firmware/host/test_footprint_baseline.py firmware/host/test_compiler_smoke.py firmware/host/test_graph_passes.py firmware/host/test_reference_interpreter.py firmware/host/test_isa_simulator.py firmware/host/test_batched_gemm.py firmware/host/test_batched_gemm_rtl_artifact.py firmware/host/test_batched_gemm_rtl_sweep.py firmware/host/test_blocked_fc_requant.py firmware/host/test_real_model_accelerator.py firmware/host/test_systolic_characterization.py firmware/host/test_rtl_sim_artifact.py firmware/host/test_rtl_residual_sim_artifact.py firmware/host/test_mnist_utpu_demo.py firmware/host/test_transformer_integration.py firmware/host/test_cost_model_regression.py firmware/host/test_cost_model_selection.py firmware/host/test_fusion_benchmark.py firmware/host/test_tiling_controller.py firmware/host/test_multi_pe_sim.py firmware/host/test_real_model_ops.py firmware/host/test_real_model_end_to_end.py firmware/host/test_phase4_isa_widen.py firmware/host/test_scheduler_allocator.py firmware/host/test_cublas_baseline.py firmware/host/test_cost_model_heldout.py firmware/host/test_selection_ab.py firmware/host/test_board_fit_audit.py firmware/host/test_scheduler_rtl_crosscheck.py firmware/host/test_scheduler_rtl_crosscheck_bigmlp.py firmware/host/test_p4_2_vivado_reports.py firmware/host/test_diff_oracle.py firmware/host/test_region_fusion.py firmware/host/test_megakernel_benchmark.py firmware/host/test_fuzzer.py firmware/host/test_egraph.py firmware/host/test_latency_analysis.py firmware/host/test_nsight_compute_profile.py -v

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

sim-iverilog-batched:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/test_vectors build/reports
	$(PYTHON) firmware/host/generate_batched_gemm_rtl_vectors.py > /dev/null
	"$(IVERILOG)" -g2012 -DICARUS -DDUMP_VCD \
		-o "$(SIM_IVERILOG_OUT)/tb_batched_gemm.out" \
		rtl/tb/tb_batched_gemm.sv $(RTL_STUBS) $(RTL_DESIGN)
	"$(VVP)" "$(SIM_IVERILOG_OUT)/tb_batched_gemm.out"
	@echo "[sim-iverilog-batched] VCD: $(SIM_IVERILOG_OUT)/tb_batched_gemm.vcd"

sim-iverilog-all: sim-iverilog-fused sim-iverilog-perf sim-iverilog-batched sim-iverilog-phase4-widen sim-iverilog-scheduler-cross-check sim-iverilog-latency

# Task 4: deterministic-latency RTL data-independence sweep. Drives
# rtl/tb/tb_latency_determinism.sv across 5 adversarial input
# distributions, asserts RTL cycle variance == 0 across them, and
# populates bench/results/latency_determinism.json::data_independence
# rtl_cycle_invariant=true. This target runs the full iverilog path
# (5 compile+run trials, ~10s); `make repro-host` uses the stub-mode
# (--skip-iverilog) regen for cross-platform CI.
sim-iverilog-latency:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/test_vectors build/reports bench/results
	$(PYTHON) firmware/host/run_latency_determinism.py \
		--output bench/results/latency_determinism.json

# Phase 7 remediation P4.1: scheduler RTL cycle cross-check.
# Generates two programs (naive + scheduled) for (M=32, K=32) and
# verifies (a) RTL_scheduled_cycles < RTL_naive_cycles, (b) the
# RTL's per-mille cycle reduction is within ±20 permille (±2.0%) of
# the simulator's per-mille reduction (the headline 4.67%-style
# claim), and (c) the scheduler's RTL invariant
# RTL_naive_bytes === RTL_scheduled_bytes holds. The Python wrapper
# `run_scheduler_rtl_crosscheck.py` does the same and also emits
# bench/results/scheduler_rtl_crosscheck.json. PROG_DEPTH stays at the
# shipping default of 1024; BUFFER_SIZE stays at 512. VCD dumping is
# off by default for this target (the program runs for ~2 ms of sim
# time, which generates a ~150 MB VCD with $dumpvars(0,*) — opt in
# with `make sim-iverilog-scheduler-cross-check DUMP_VCD=1` if you
# want it).
DUMP_VCD ?= 0
SCHEDULER_CROSS_DEFINES = -DICARUS
ifeq ($(DUMP_VCD),1)
SCHEDULER_CROSS_DEFINES += -DDUMP_VCD
endif

sim-iverilog-scheduler-cross-check:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/test_vectors build/reports
	$(PYTHON) firmware/host/generate_scheduler_rtl_test_vectors.py > /dev/null
	"$(IVERILOG)" -g2012 $(SCHEDULER_CROSS_DEFINES) \
		-o "$(SIM_IVERILOG_OUT)/tb_scheduler_cycles.out" \
		rtl/tb/tb_scheduler_cycles.sv $(RTL_STUBS) $(RTL_DESIGN)
	"$(VVP)" "$(SIM_IVERILOG_OUT)/tb_scheduler_cycles.out"
	@echo "[sim-iverilog-scheduler-cross-check] artifact: bench/results/scheduler_rtl_crosscheck.json (run run_scheduler_rtl_crosscheck.py)"

# Phase 4 widened RTL sim: INT8 datapath, ARRAY_SIZE=8, BUFFER_SIZE=4096,
# EXT_ADDR_EN=1 (2-word address layout). Compares against the python
# isa_simulator oracle (which is itself NumPy-validated).
sim-iverilog-phase4-widen:
	mkdir -p "$(SIM_IVERILOG_OUT)" build/test_vectors build/reports
	$(PYTHON) firmware/host/generate_phase4_widen_rtl_vectors.py > /dev/null
	"$(IVERILOG)" -g2012 -DICARUS -DDUMP_VCD \
		-o "$(SIM_IVERILOG_OUT)/tb_phase4_widen.out" \
		rtl/tb/tb_phase4_widen.sv $(RTL_STUBS) $(RTL_DESIGN)
	"$(VVP)" "$(SIM_IVERILOG_OUT)/tb_phase4_widen.out"
	@echo "[sim-iverilog-phase4-widen] VCD: $(SIM_IVERILOG_OUT)/tb_phase4_widen.vcd"

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
	@echo "[repro-host] batched GEMM correctness (Phase 0) -> bench/results/batched_gemm_correctness.json"
	$(PYTHON) firmware/host/run_batched_gemm_correctness.py
	@echo "[repro-host] systolic characterization (Phase 1; stub on hosts without iverilog) -> bench/results/systolic_characterization.json"
	$(PYTHON) firmware/host/run_systolic_characterization.py
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
	@echo "[repro-host] tiling correctness (Phase 3) -> bench/results/tiling_correctness.json"
	$(PYTHON) firmware/host/run_tiling_correctness.py
	@echo "[repro-host] scheduler cycles (Phase 5) -> bench/results/scheduler_cycles.json"
	$(PYTHON) firmware/host/run_scheduler_benchmark.py
	@echo "[repro-host] cost-model held-out generalization (Phase 7) -> bench/results/cost_model_heldout.json"
	$(PYTHON) firmware/host/run_cost_model_heldout.py
	@echo "[repro-host] cuBLAS baseline schema/stub (Phase 7) -> bench/results/cublas_baseline.json"
	$(PYTHON) firmware/host/run_cublas_baseline.py
	@echo "[repro-host] cuBLAS baseline recompute aggregate (Phase 7 v1.2; surfaces cuBLASLt IMMA INT8 GEMM gap slots WITHOUT re-running CUDA) -> bench/results/cublas_baseline.json"
	-$(PYTHON) firmware/host/run_cublas_baseline.py --recompute-aggregate-only || true
	@echo "[repro-host] selection A/B schema/stub (Phase 7 remediation P2.2) -> bench/results/selection_ab.json"
	$(PYTHON) firmware/host/run_selection_ab.py
	@echo "[repro-host] board-fit audit (Phase 7 remediation P3) -> bench/results/board_fit_audit.json"
	$(PYTHON) firmware/host/run_board_fit_audit.py
	@echo "[repro-host] scheduler RTL cross-check (Phase 7 remediation P4.1; stub on hosts without iverilog) -> bench/results/scheduler_rtl_crosscheck.json"
	-$(PYTHON) firmware/host/run_scheduler_rtl_crosscheck.py || true
	@echo "[repro-host] megakernel payoff stub (Task 1; populated on CUDA hosts via repro-cuda) -> bench/results/megakernel_payoff.json"
	$(PYTHON) firmware/host/run_megakernel_benchmark.py
	@echo "[repro-host] megakernel recompute aggregate (Task 1 v1.1; surfaces launch-count reduction + cuda_graphs_op_by_op arm slots WITHOUT re-running CUDA) -> bench/results/megakernel_payoff.json"
	$(PYTHON) firmware/host/run_megakernel_benchmark.py --recompute-aggregate-only --output bench/results/megakernel_payoff.json
	@echo "[repro-host] Nsight Compute occupancy/bottleneck profile stub (Task 1 v1.1; populated on WSL2 + CUDA via 'make nsight-compute-profile') -> bench/results/nsight_compute_profile.json"
	$(PYTHON) firmware/host/run_nsight_compute_profile.py --skip-ncu
	@echo "[repro-host] superopt payoff (Task 3 equality-saturation; 4 planted + 512 random graphs; isa_cycle_model) -> bench/results/superopt_payoff.json"
	$(PYTHON) firmware/host/run_superopt_benchmark.py --num-random-graphs 512 --seed-start 0 --cost-function isa_cycle_model --output bench/results/superopt_payoff.json
	@echo "[repro-host] fuzzer report ci_seeded (Task 2; cuda_megakernel relation skips on non-CUDA hosts) -> bench/results/fuzzer_report.json"
	$(PYTHON) firmware/host/run_fuzzer.py --mode ci_seeded --seed 1234 --num-graphs 64
	@echo "[repro-host] latency determinism (Task 4; static-only stub on hosts without iverilog) -> bench/results/latency_determinism.json"
	$(PYTHON) firmware/host/run_latency_determinism.py --skip-iverilog
	@echo "[repro-host] done. CUDA-only artifacts (ResNet-18, populated cuBLAS baseline, populated selection A/B, populated megakernel payoff) regenerate via 'make repro-cuda'."

repro-cuda:
	mkdir -p bench/results
	@echo "[repro-cuda] ResNet-18 end-to-end (requires CUDA + Linux/WSL2) -> bench/results/real_model_end_to_end.json"
	$(PYTHON) firmware/host/run_real_model_end_to_end.py --output bench/results/real_model_end_to_end.json
	$(PYTHON) -m pytest firmware/host/test_real_model_ops.py firmware/host/test_real_model_end_to_end.py -q
	@echo "[repro-cuda] cuBLAS baseline (Phase 7 v1.2; populated with cuBLAS GEMV + cuBLASLt IMMA INT8 GEMM + Inductor arms) -> bench/results/cublas_baseline.json"
	$(PYTHON) firmware/host/run_cublas_baseline.py --warmup 10 --iters 50
	@echo "[repro-cuda] selection A/B (Phase 7 remediation P2.2, populated) -> bench/results/selection_ab.json"
	$(PYTHON) firmware/host/run_selection_ab.py --warmup 10 --iters 50
	@echo "[repro-cuda] megakernel payoff (Task 1 + v1.1: 6 workloads incl. 1024^2 / 4096^2 arith-bound; 4 arms incl. cuda_graphs_op_by_op) -> bench/results/megakernel_payoff.json"
	$(PYTHON) firmware/host/run_megakernel_benchmark.py --warmup 10 --iters 50
	@echo "[repro-cuda] Nsight Compute occupancy/bottleneck profile (Task 1 v1.1, populated) -> bench/results/nsight_compute_profile.json"
	$(PYTHON) firmware/host/run_nsight_compute_profile.py

repro: repro-host
	@echo "[repro] host artifacts regenerated. For ResNet-18 + populated cuBLAS baseline (CUDA), run: make repro-cuda"

# ---------------------------
# Fuzzer (Task 2 / `utpu_upgrade_plan.md` §4).
# `fuzz`           : short deterministic ci_seeded gate (regenerates fuzzer_report.json
#                    with 64 seeds, no minimization writes — same as repro-host).
# `fuzz-discovery` : long discovery run (10k seeds), commits any minimized real-bug repros
#                    to firmware/host/fuzz/repros/. Honest claim: zero real bugs found in v1
#                    development; running discovery is how new bugs would surface.
# ---------------------------
fuzz:
	mkdir -p bench/results
	$(PYTHON) firmware/host/run_fuzzer.py --mode ci_seeded --seed 1234 --num-graphs 64

superopt:
	mkdir -p bench/results
	$(PYTHON) firmware/host/run_superopt_benchmark.py --num-random-graphs 512 --seed-start 0 --cost-function isa_cycle_model --output bench/results/superopt_payoff.json

fuzz-discovery:
	mkdir -p bench/results firmware/host/fuzz/repros
	$(PYTHON) firmware/host/run_fuzzer.py --mode discovery --seed 1234 --num-graphs 10000 --write-repros

# Task 4 — deterministic-latency static analysis standalone target.
# Stub mode (no iverilog needed) for cross-platform repro.
# For the full RTL data-independence arm, run `make sim-iverilog-latency`.
latency-determinism:
	mkdir -p bench/results
	$(PYTHON) firmware/host/run_latency_determinism.py --skip-iverilog \
		--output bench/results/latency_determinism.json

# Task 1 v1.1 — recompute the aggregate block of megakernel_payoff.json from
# the existing per-arm timings + launch counts WITHOUT re-running CUDA. Use
# when the aggregate schema changed (e.g. launch-count fields added) but the
# raw per-arm data is still current. For a full re-run from scratch on a
# CUDA host, use `make repro-cuda`.
megakernel-recompute-aggregate:
	$(PYTHON) firmware/host/run_megakernel_benchmark.py --recompute-aggregate-only \
		--output bench/results/megakernel_payoff.json

# Phase 7 v1.2 — recompute the aggregate block of cublas_baseline.json from
# the existing per-shape timings WITHOUT re-running CUDA. Surfaces the
# cuBLASLt IMMA INT8 GEMM gap slots (gap_vs_cublaslt_int8_pct_median) on
# legacy artifacts; on a host without live IMMA data the IMMA gap fields
# collapse to None (no fabrication). For a full re-run from scratch on a
# CUDA host, use `make repro-cuda`.
cublas-baseline-recompute-aggregate:
	$(PYTHON) firmware/host/run_cublas_baseline.py --recompute-aggregate-only \
		--output bench/results/cublas_baseline.json

# Task 1 v1.1 — Nsight Compute occupancy / bottleneck profile of the
# fused_region kernel on the 4 locked launch-bound workloads. Requires
# `ncu` (Nsight Compute CLI) on PATH and a CUDA device. On hosts without
# ncu, emits a stub artifact via `--skip-ncu` (the same path repro-host
# uses).
nsight-compute-profile:
	mkdir -p bench/results
	$(PYTHON) firmware/host/run_nsight_compute_profile.py \
		--output bench/results/nsight_compute_profile.json
