# PROJECT_CONTEXT

## Identity
`uTPU` is a scoped blocked-FC MLP compiler/runtime project with a pass-based Graph IR pipeline and differential validation against a deterministic NumPy oracle.

## Compiler Path
```text
torch.fx trace
  -> fx_importer.py
  -> graph_ir.py
  -> graph_passes.py
      shape_inference
      linear_relu_fusion
      dead_code_elimination
      memory_planning
      backend_legality
  -> graph_lowering.py
  -> backend_lowering.py
      -> cuda_blocked_fc_backend.py
      -> lowering_blocked_fc_utpu.py
```

## Correctness Path
- Oracle: `graph_reference_interpreter.py` (NumPy-only execution).
- Harness: `differential_test_harness.py` writes `build/reports/differential_test_report.json`.
- ISA/RTL bitmatch: `run_isa_rtl_bitmatch.py` compares Python ISA simulator fetch bytes with Verilog RTL simulation output for compiled fused programs.

## CUDA Autotuner Path
- Cost model: `cost_model.py`.
- Calibration/refit: `calibrate_cost_model.py`.
- Pruned replay evaluation: `evaluate_pruned_autotuner.py`.
- Current supported claim: measured-data replay prunes 16 schedules to `4.92` profiled candidates on average with max replay regression under `0.5%`.
- Live small-shape smoke artifact exists at `build/reports/live_autotuner_comparison.json`, but replay remains the primary quality evidence.

## CI / Regression Path
- Workflow: `.github/workflows/ci.yml`.
- Current GitHub Actions state: green on `main` at commit `75a626d`, run `26256684852`.
- CI runs a narrow host regression set covering footprint baseline, compiler smoke, graph passes, reference interpreter, ISA simulator, and optional RTL artifact checks.
- Optional artifact tests skip when clean CI lacks local-only files such as `software/model/weights` or RTL metrics reports.

## Validation Boundaries
- CUDA backend uses compiled execution when available.
- uTPU backend comparison uses software quantized emulation in this repo path (no board runtime execution).
- Supported scope remains blocked-FC MLP subset only.
