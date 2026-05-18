# RESUME_CLAIMS.md

## Resume-Safe Claims (Current)
- Built a scoped PyTorch compiler for blocked-FC MLP inference with a pass-based Graph IR stage: shape inference, Linear+ReLU fusion, dead-code elimination, liveness-driven memory planning, and backend legality checks.
- Implemented a deterministic NumPy Graph IR interpreter as an independent correctness oracle.
- Added a differential harness that compares backend outputs to the oracle on fixed MLP shapes and emits a reproducible JSON report.
- Verified bit-accurate agreement between a Python uTPU ISA simulator and Verilog RTL on two compiled fused MLP programs.

## Candidate Claims (Cost Model)
- Built a CUDA-event calibration harness and CTA-working-set memory latency model for an INT4 blocked-FC CUDA kernel; fresh calibration over 3,072 shape/schedule measurements reached log_R2 `0.9323`, MAPE `10.68%`, and requested 80/20 shape-grid holdout MAPE `10.11%`.
- Integrated the cost model into the CUDA schedule autotuner as a conservative pruning policy; current calibrated replay profiles `4.92` of 16 schedules on average (`3.25x` search reduction), with all selected schedules within `1%` of exhaustive best and max replay regression under `0.5%`.
- Added liveness-analysis memory planning with greedy buffer reuse; sample memory-plan artifact reduces persistent activation allocation from `48` to `32` bytes (`33.33%`).
- Candidate wording after TorchInductor is rerun on a supported platform: "validated against a NumPy reference interpreter, TorchInductor output, and CUDA/uTPU backends via differential testing."

## Evidence
- Pass pipeline: `firmware/host/graph_passes.py`, `firmware/host/test_graph_passes.py`
- Oracle: `firmware/host/graph_reference_interpreter.py`, `firmware/host/test_reference_interpreter.py`
- Differential report: `firmware/host/differential_test_harness.py`, `firmware/host/test_differential_harness.py`, `build/reports/differential_test_report.json`
- ISA/RTL bitmatch: `firmware/host/isa_simulator.py`, `firmware/host/run_isa_rtl_bitmatch.py`, `firmware/host/test_isa_simulator.py`, `build/reports/isa_rtl_bitmatch_report.json`
- Cost model candidate: `firmware/host/cost_model.py`, `firmware/host/calibrate_cost_model.py`, `build/reports/cost_model_calibration.json`
- Holdout validation: `build/reports/cost_model_holdout_validation.json`, `build/reports/cost_model_holdout_validation.md`
- Pruned autotuner: `firmware/host/cuda_autotuner.py`, `firmware/host/evaluate_pruned_autotuner.py`, `build/reports/pruned_autotuner_report.json`
- Memory planning: `firmware/host/graph_passes.py`, `firmware/host/generate_memory_plan_report.py`, `build/reports/memory_plan_report.json`

## Caveat To Keep
- uTPU comparison is software quantized emulation in this harness path, not physical board execution.
- TorchInductor oracle is implemented but skipped in the current Windows artifact with `WinError 50`; do not claim passing TorchInductor validation until rerun on a supported Linux/WSL stack.
- ISA/RTL bitmatch is simulation evidence across two current compiled programs, not board execution and not broad random-program verification.
- Cost-model claim should be scoped to calibrated/measured shape-grid interpolation. A stricter actual-layer-shape holdout has p95 `37.85%` with structured residuals at `(in=64,out=512,tpb=256)`, so do not imply broad unseen-shape prediction without more evidence.
- Pruned-autotuner quality claim is measured-data replay over the calibrated grid. A small live CUDA smoke check exists, but tiny kernel timing noise means it should not replace replay as the primary quality evidence.
