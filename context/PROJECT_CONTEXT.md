# PROJECT_CONTEXT.md

## Purpose
`uTPU` is a scoped ML compiler/runtime for blocked-FC MLP inference with a pass-based Graph IR pipeline, dual lowering paths (CUDA/uTPU), and a differential-validation harness.

## Data Flow / Architecture Path
PyTorch module -> `torch.fx` trace -> FX import -> Graph IR -> pass pipeline -> blocked-FC lowering -> backend lowering -> execution/emulation -> differential report artifact.

## Core Modules
- `firmware/host/graph_ir.py`: Graph IR kinds (`linear`, `linear_relu`, `relu`, `add`, `view`).
- `firmware/host/graph_passes.py`: explicit passes:
  - `shape_inference`
  - `linear_relu_fusion`
  - `dead_code_elimination`
  - `memory_planning`
  - `backend_legality`
- `firmware/host/graph_reference_interpreter.py`: deterministic NumPy reference oracle.
- `firmware/host/differential_test_harness.py`: fixed-shape backend-vs-reference harness and JSON report writer.
- `firmware/host/cuda_autotuner.py`: CUDA schedule search with optional cost-model top-K pruning.
- `firmware/host/pytorch_compiler.py`: compile entrypoint integrating passes and interpreter access.
- `firmware/host/isa_simulator.py`: Python simulator for RTL-observable uTPU ISA behavior.
- `firmware/host/run_isa_rtl_bitmatch.py`: compares Python ISA simulator fetch bytes against RTL simulation fetch bytes.

## Evidence Status
- Strongly supported:
  - Pass pipeline behavior and legality errors via unit/integration tests.
  - Deterministic reference interpreter correctness on hand-computed graph tests.
  - Differential report artifact generated at `build/reports/differential_test_report.json`.
  - Differential fixtures are non-identity and include signed inputs/weights with tiny deterministic jitter.
  - CUDA cost model has fresh calibration artifact at `build/reports/cost_model_calibration.json`.
  - Cost-model-pruned autotuner report generated at `build/reports/pruned_autotuner_report.json`; current replay profiles `4.92` of 16 candidates on average with `0.49%` max quality regression.
  - Small live CUDA autotuner smoke artifact generated at `build/reports/live_autotuner_comparison.json`; use replay as the primary quality claim.
  - Liveness memory-planning artifact generated at `build/reports/memory_plan_report.json`.
  - Python ISA simulator bitmatches RTL fetch bytes for two compiled fused programs in `build/reports/isa_rtl_bitmatch_report.json`.
  - README front page now links to tracked evidence summary at `docs/EVIDENCE.md`.
- Scoped caveat:
  - uTPU differential path currently runs quantized runtime emulation (`quantized_reference_emulation`), not physical board execution.
  - TorchInductor oracle is implemented but currently skipped on this Windows run with `WinError 50`.

## Explicit Non-Goals / Risky Claims
- Not a general PyTorch compiler.
- Not transformer support.
- Not physical board execution evidence.

## Key Commands
- `python -m pytest firmware/host/test_graph_passes.py -q`
- `python -m pytest firmware/host/test_reference_interpreter.py -q`
- `python -m pytest firmware/host/test_differential_harness.py -q`

## Resume-Safe Summary
The repo now has a pass-based Graph IR compiler stage, liveness memory planning, a differential-validation harness against a deterministic NumPy oracle, and measured CUDA cost-model/autotuner artifacts for the blocked-FC subset.
It also has Python ISA simulator vs Verilog RTL bitmatch evidence for the current fused compiled-program path.
