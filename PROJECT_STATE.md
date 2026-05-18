# PROJECT_STATE.md

Last updated: 2026-05-18

## Current Scope
Blocked-FC MLP subset only (`Linear`, `ReLU`, fused `Linear+ReLU`) with pass-based IR transformation and CUDA/uTPU lowering.

## Architecture Snapshot
```text
PyTorch FX trace
  -> Graph IR import
  -> Pass pipeline
      1) shape_inference
      2) linear_relu_fusion
      3) dead_code_elimination
      4) memory_planning
      5) backend_legality
  -> blocked-FC lowering
  -> backend execution path
  -> differential harness report
```

## Landed Deliverables
- A: pass manager + pass units + pass pipeline dump.
- B: deterministic NumPy Graph IR reference interpreter.
- C: differential harness + pytest gate + `build/reports/differential_test_report.json`.
- D: Python ISA simulator / Verilog RTL fetch-byte bitmatch evidence for two compiled fused programs.
- E: CUDA cost-model-guided autotuner replay evidence for blocked-FC schedule pruning.

## Differential Validation Status
- Shape set: `(4,8,4)`, `(8,16,8)`, `(16,32,16)`.
- Tolerance: `atol=1e-5`, `rtol=1e-5`.
- CUDA: compiled path compared to reference; skips cleanly when unavailable.
- uTPU: compared via `quantized_reference_emulation` (software), not board execution.
- Fixtures: signed sparse integer weights and signed inputs (non-identity), chosen to stay quantization-stable for deterministic backend/reference agreement.

## CUDA Autotuner Status
- Current replay artifact: `build/reports/pruned_autotuner_report.json`.
- Policy profiles `4.92` of 16 schedules on average (`3.25x` search reduction).
- Replay max quality regression is `0.49%`; all selected schedules are within `1%` of exhaustive best.
- Live smoke artifact: `build/reports/live_autotuner_comparison.json`; use it as execution/wall-clock smoke evidence, not the primary quality claim.

## Boundaries
- Not a general PyTorch compiler.
- No transformer support.
- No physical board-execution proof.
