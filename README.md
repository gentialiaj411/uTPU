# uTPU: Scoped ML Compiler + CUDA/uTPU Backend Lab

[![CI](https://github.com/gentialiaj411/uTPU/actions/workflows/ci.yml/badge.svg)](https://github.com/gentialiaj411/uTPU/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/badge/Evidence_Explorer-live-0A7EA4)](https://gentialiaj411.github.io/uTPU/)

![Compiler pipeline terminal preview](docs/inspect_compiler_pipeline_demo.svg)

**Evidence Explorer (live):** [https://gentialiaj411.github.io/uTPU/](https://gentialiaj411.github.io/uTPU/) — tier-fenced claims rebuilt from committed `bench/results/` artifacts.

For a full top-down tour and interview prep, see [WALKTHROUGH.md](WALKTHROUGH.md). Resume-oriented claim inventory: [RESUME_CLAIMS.md](RESUME_CLAIMS.md).

`uTPU` is a focused ML systems project: PyTorch models lower into a custom Graph IR, then into generated CUDA kernels and/or custom uTPU ISA/RTL programs. The point is not broad framework coverage. The point is compiler passes, backend lowering, measurement discipline, autotuning, and hardware-style verification on narrow, explicit workload families.

## Scope

Supported:

- Batch-1 MLP-style `Linear -> ReLU -> Linear` flows.
- Single-block transformer path: `Q/K/V linear -> attention -> output projection -> residual -> RMSNorm -> (optional MLP reuse)`.
- Pass-based Graph IR: shape inference, Linear+ReLU fusion, dead-code elimination, liveness memory planning, backend legality.
- Transformer pass pipeline: attention decomposition (`scaled_dot_product_attention` -> `permute + batched_matmul + scale + softmax + batched_matmul`) plus `scale+softmax` fusion.
- INT4 blocked fully connected lowering for CUDA and uTPU program emission.
- CUDA runtime execution through generated NVRTC kernels.
- uTPU ISA generation plus Python ISA simulation and Verilog RTL simulation evidence; Vivado post-route timing/util on Artix-7 A7-100T (`xc7a100tcsg324-1`).
- `torch.compile` custom backend (`backend="utpu"`) that dispatches supported FX subgraphs through the existing CUDA / uTPU ISA pipeline and falls back to eager for unsupported ops.
- **ResNet-18 (CUDA graph path):** FX import for `Conv2d`, `BatchNorm2d` (folded into conv), `MaxPool2d`, `AdaptiveAvgPool2d`, residual `add`, and `Linear`; end-to-end compile + execution through the Graph IR runtime. Evidence: [bench/results/real_model_end_to_end.json](bench/results/real_model_end_to_end.json), `firmware/host/test_real_model_end_to_end.py`.

Not claimed:

- General PyTorch compiler support beyond the documented families (blocked-FC MLP, transformer blocks, ResNet-18 on CUDA).
- Full GPT/BERT graph coverage.
- Production `torch.compile` backend.
- End-to-end speedup over PyTorch/cuBLAS on tiny benchmarks (or FPGA speedup vs GPU).
- On-board FPGA / silicon execution (P0 open) — all RTL cycle and UART claims below are iverilog or post-route synth, not board capture.
- Shipping at the demonstrated 100 MHz ceiling (marginal WNS); the ship point is ~83.3 MHz.
- uTPU cycle-model selection regret comparable to CUDA’s 16-candidate menu (uTPU menu is 5 candidates).
- Concurrent LOAD+COMPUTE as the next win (isolated-GEMM attribution parks that path).
- Part A descriptor ISA (parked).

## Evidence-Backed Claims

### Hardware / RTL (Artix-7 path)

- **DSP 192 → 72 and clock shipping:** requant/ReLU finalize rightsizing (Step1+2) cuts Artix-7 A7-100T DSP from **192/240** to **72/240** on `ARRAY_SIZE=8` INT8. Design-space shipping close is **~83.3 MHz** (`12 ns`, post-route **WNS=+0.271 ns**, margin thin) on `xc7a100tcsg324-1`; demonstrated Fmax ceiling is **100 MHz** (`10 ns`, **WNS=+0.012 ns**, marginal — cite WNS if quoted). Workload/harness: N=8 INT8, `MAX_BATCH_COUNT=48`, Vivado P&R. Evidence: [bench/results/requant_rightsizing_synth.json](bench/results/requant_rightsizing_synth.json), [bench/results/design_space_sweep.json](bench/results/design_space_sweep.json), [docs/HARDWARE_DESIGN_SPACE.md](docs/HARDWARE_DESIGN_SPACE.md).

- **Instruction-stream attribution:** on isolated GEMM **64×64 B=48 N=16** (fast-UART TB), instruction-stream is **66%** of on-chip core cycles and buffer→PE **LOAD is 1.1%**; compute ~5.7%. PE-level concurrent LOAD+COMPUTE cannot materially help under this harness. Evidence: [bench/results/cycle_attribution.json](bench/results/cycle_attribution.json).

- **BSTORE write-arm widening:** `BSTORE_WIDTH=8` yields **~1.89×** end-to-end cycle reduction on multi-layer fused MNIST (pre-widen 6523 → post-widen 3445 cycles). Evidence: [bench/results/bstore_path_measure.json](bench/results/bstore_path_measure.json), [bench/results/cycle_attribution_mnist.json](bench/results/cycle_attribution_mnist.json).

- **Steady-state compute share:** with buffer-resident weights (A5 fill + remapped packing), fused-MNIST compute share rises from cold **~20.6%** to steady-state **~56.2%** (bit-exact vs cold oracle; mean 1261 cycles vs cold 3445). Evidence: [bench/results/steady_state_attribution.json](bench/results/steady_state_attribution.json).

- **Instruction capacity:** historical `PROG_DEPTH=1024` admits **4/14** board-fit shapes; shipping **`PROG_DEPTH=65536`** admits **10/14** (`artix_a7100t_bram_max`). Evidence: [bench/results/board_fit_audit.json](bench/results/board_fit_audit.json), [bench/results/prog_depth_sweep.json](bench/results/prog_depth_sweep.json).

- **uTPU cycle cost model:** held-out **log R² 0.924**, **MAPE 10.04%**, **zero selection regret** over a **5-candidate** `(batch, hoist)` space (contrast CUDA held-out mean regret **5.21%** on a **16-candidate** space). Evidence: [bench/results/utpu_cycle_model_heldout.json](bench/results/utpu_cycle_model_heldout.json), [docs/COSTMODEL_COMPARISON.md](docs/COSTMODEL_COMPARISON.md), [bench/results/cost_model_heldout.json](bench/results/cost_model_heldout.json).

- **Determinism vs GPU:** FPGA/RTL cycle variance **0** across adversarial+random inputs; at shipping **~83 MHz** median wall-clock latency is **~1.89×** the GPU p50 (loss, not speedup). FPGA p99 equals p50 (flat); GPU p99 / p99.9 tails are recorded in the artifact. Evidence: [bench/results/latency_determinism_vs_gpu.json](bench/results/latency_determinism_vs_gpu.json), [bench/results/latency_determinism.json](bench/results/latency_determinism.json), [docs/latency_determinism_vs_gpu_logx.png](docs/latency_determinism_vs_gpu_logx.png).

### Compiler / CUDA / simulation

- Compiler correctness: differential harness compares the NumPy Graph IR interpreter, CUDA compiled backend, uTPU quantized emulation, and a TorchInductor oracle when the local platform supports it. Evidence: [docs/EVIDENCE.md#differential-testing](docs/EVIDENCE.md#differential-testing), [firmware/host/differential_test_harness.py](firmware/host/differential_test_harness.py).

- Real-model CUDA path: ResNet-18 lowers end-to-end through FX → Graph IR → graph-op execution. Parity vs eager PyTorch at `rtol=1e-3`, `atol=1e-3`. Evidence: [bench/results/real_model_end_to_end.json](bench/results/real_model_end_to_end.json).

- CUDA performance engineering: calibrated CUDA-event data backs a schedule-aware analytical cost model and pruning policy for blocked-FC autotuning. Measured-data replay profiles `4.92` of 16 schedules on average (`3.25x` search reduction), keeps every selected schedule within `1%` of exhaustive best, and lowers max replay regression from strict top-k's `5.90%` to `0.49%`. Evidence: [docs/EVIDENCE.md#cost-model-and-pruned-autotuner](docs/EVIDENCE.md#cost-model-and-pruned-autotuner), [firmware/host/cost_model.py](firmware/host/cost_model.py), [firmware/host/cuda_autotuner.py](firmware/host/cuda_autotuner.py).

- Hardware verification (simulation): Python ISA simulator and Verilog RTL produce bit-identical fetch bytes on compiled fused MLP programs. Evidence: [docs/EVIDENCE.md#python-isa-simulator--rtl-bitmatch](docs/EVIDENCE.md#python-isa-simulator--rtl-bitmatch), [firmware/host/isa_simulator.py](firmware/host/isa_simulator.py), [rtl/tb/tb_fused_compressed_program.sv](rtl/tb/tb_fused_compressed_program.sv).

- Cost-model **generalization on unseen shapes** (CUDA): held-out `log_R^2 = 0.926`, `MAPE = 14.32%`, selection-regret mean `5.21%` / max `11.90%`. Evidence: [bench/results/cost_model_heldout.json](bench/results/cost_model_heldout.json).

- Board-fit audit: per-config which workloads fit instruction BRAM (`pynqz2_baseline` 4/14, `pynqz2_bram_max` 8/14, `artix_a7100t_bram_max` 10/14, `vu13p_uram` 12/14). Evidence: [bench/results/board_fit_audit.json](bench/results/board_fit_audit.json).

- Cost-model schedule is consumed at CUDA runtime (`schedule_source="cost_model"`). Wall-clock A/B percentages on unlocked WSL2 remain `[needs-locked-clock-artifact]`. Evidence: [bench/results/selection_ab.json](bench/results/selection_ab.json).

- Full-stack writeup + repro: [docs/WRITEUP.md](docs/WRITEUP.md), [docs/REPRO.md](docs/REPRO.md).

- cuBLAS / Inductor baseline methodology with dtype caveats; published GPU %-gaps on unlocked clocks remain `[needs-locked-clock-artifact]`. Evidence: [bench/results/cublas_baseline.json](bench/results/cublas_baseline.json).

- CUDA fusion benchmark: no GPU fusion speedup claimed at tiny shapes; %-delta `[needs-locked-clock-artifact]`. Evidence: [bench/results/fusion_payoff.json](bench/results/fusion_payoff.json).

- Scheduler RTL cycle cross-check: sim **4.67%** cycle reduction reproduced at RTL percentage level (±2.0% permille tolerance) with bit-exact naive vs scheduled fetch bytes. Evidence: [bench/results/scheduler_rtl_crosscheck.json](bench/results/scheduler_rtl_crosscheck.json), [rtl/tb/tb_scheduler_cycles.sv](rtl/tb/tb_scheduler_cycles.sv).

## Architecture

```text
PyTorch module
  -> torch.fx trace
  -> custom Graph IR
  -> pass pipeline
       shape inference
       Linear+ReLU fusion
       dead-code elimination
       liveness memory planning
       backend legality
  -> blocked-FC + transformer-op lowering
  -> backend lowering
       CUDA: NVRTC kernel + schedule autotuner + cost model
       uTPU: ISA program words + Python ISA sim + Verilog RTL sim
```

## Fast Demo

```bash
python examples/inspect_compiler_pipeline.py
```

This prints the FX graph, post-pass Graph IR, blocked-FC schedule, CUDA kernel metadata, uTPU instruction footprint, fallback lists, and writes `build/reports/compiler_introspection_tiny_mlp.json`.

Expected summary:

```text
FX graph: x -> fc1 -> relu -> fc2 -> output
Graph IR ops: linear_relu, linear
fallback_ops=[]
CUDA backend: blocked_fc_int4_kernel executable=True/False depending on local CUDA driver bindings
uTPU ISA footprint: total_utpu_instruction_words=434
```

## Visual Compiler Pipeline Demo

```bash
python examples/visualize_compiler_pipeline.py
```

This emits:

- terminal summary from the actual compile run
- `build/reports/compiler_pipeline_visual.json`
- `build/reports/compiler_pipeline_visual.html`

## Reproduce The Key Evidence

```bash
python -m pytest firmware/host/test_differential_harness.py -q
python -m pytest firmware/host/test_transformer_integration.py -q
python -m pytest firmware/host/test_isa_simulator.py -q
python firmware/host/run_isa_rtl_bitmatch.py --output-json build/reports/isa_rtl_bitmatch_report.json --output-md build/reports/isa_rtl_bitmatch_report.md
python firmware/host/evaluate_pruned_autotuner.py --top-k 4 --output-json build/reports/pruned_autotuner_report.json
python examples/inspect_compiler_pipeline.py
```

Rebuild the Evidence Explorer data bundle locally:

```bash
python tools/build_frontend_data.py
cd frontend && npm install && npm run build
```

For CUDA calibration and holdout validation, use the calibration scripts under `firmware/host/`; these generate local reports under `build/reports/`.

Current GitHub Actions status is reflected by the badge above. For the latest public details, check the GitHub Actions history for `ci.yml` and `pages.yml`.

## Resume-Safe Wording

- Built a pass-based Graph IR compiler for a scoped MLP subset with shape inference, Linear+ReLU fusion, liveness memory planning, backend legality checks, and differential testing against reference/backend oracles.

- Built a cost-model-guided CUDA blocked-FC autotuner that prunes a 16-candidate schedule space to `4.92` measured candidates on calibrated replay (`3.25x` reduction), with all selected schedules within `1%` of exhaustive best and max replay regression under `0.5%`.

- Closed Artix-7 timing at ~83 MHz (WNS=+0.271) after rightsizing DSP 192→72; attributed on-chip cycles (instruction-stream 66%, LOAD 1.1%); and showed zero FPGA cycle variance with ~1.89× median-latency loss vs GPU at the shipping clock.

- Verified bit-accurate agreement between a Python uTPU ISA simulator and Verilog RTL on compiled fused MLP programs.

## Important Caveats

- The TorchInductor oracle is wired into the differential harness, but Windows runs may skip when the local platform or toolchain prerequisites are missing. Rerun on a supported Linux/WSL TorchInductor stack before claiming TorchInductor pass coverage.

- The uTPU differential backend in the compiler harness is software emulation. ISA/RTL bitmatch and cycle attribution are simulation-based, not board execution.

- uTPU ISA emission remains linear-only today. Transformer ops and ResNet-18 conv/pool ops are lowered and executable in the CUDA/compiler graph path; on uTPU target they return explicit unsupported diagnostics per op.
- Transformer and ResNet graph-op execution prefer NVRTC (`cuda-python`) or Torch-CUDA when available. When neither is available, ResNet end-to-end tests use the NumPy Graph IR reference executor (see `execution_backend` in `bench/results/real_model_end_to_end.json`).
- Locked-clock GPU latency percentages remain `[needs-locked-clock-artifact]` on the published unlocked WSL2 laptop host.
