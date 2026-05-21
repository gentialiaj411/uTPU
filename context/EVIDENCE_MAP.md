# EVIDENCE_MAP.md

## Supported Claims
- FX import to Graph IR for blocked-FC MLP subset exists.
- Pass-based Graph IR compiler stage exists and runs in ordered passes:
  - `shape_inference`
  - `linear_relu_fusion`
  - `dead_code_elimination`
  - `memory_planning`
  - `backend_legality`
- Deterministic NumPy reference interpreter exists for supported IR ops.
- Differential harness exists and emits `build/reports/differential_test_report.json` with:
  - per-shape/per-backend status
  - max abs and max rel error
  - tolerance
  - environment info
  - non-identity deterministic fixtures (signed sparse weights/inputs with tiny deterministic jitter)
  - TorchInductor oracle entry when `torch.compile(..., backend="inductor")` is available
- CUDA blocked-FC cost model has fresh calibration evidence:
  - `build/reports/cost_model_calibration.json`
  - log_R2 `0.9323`, MAPE `10.68%`, p95 `25.41%`
- Cost-model-pruned autotuner has measured-data replay evidence:
  - `build/reports/pruned_autotuner_report.json`
  - policy profiles `4.92` of 16 candidates on average, `3.25x` search reduction, max quality regression `0.49%`, within-1% fraction `1.00`
  - strict top-k comparison remains in the same report for honesty
- Liveness-driven memory planning exists:
  - `memory_planning_pass`
  - `build/reports/memory_plan_report.json`
- Python ISA simulator / RTL bitmatch exists:
  - `firmware/host/isa_simulator.py`
  - `firmware/host/run_isa_rtl_bitmatch.py`
  - `build/reports/isa_rtl_bitmatch_report.json`
  - current result: `all_isa_expected_bitmatch=true`, `all_isa_rtl_bitmatch=true`
- GitHub front-page evidence summary exists:
  - `README.md`
  - `docs/EVIDENCE.md`
  - `docs/inspect_compiler_pipeline_demo.svg`
- GitHub Actions CI is green:
  - workflow: `.github/workflows/ci.yml`
  - current green run: `26256684852` on `main` at commit `75a626d`
  - clean-checkout optional artifacts skip instead of failing

## Important Caveat
- uTPU differential entry uses `quantized_reference_emulation` in software because board runtime execution is not available in this path.
- Current harness fixtures are quantization-stable; exact `0.0` errors are expected and not evidence of identity passthrough.
- Current TorchInductor oracle is wired in but skipped on Windows with `WinError 50`; do not claim TorchInductor pass validation until rerun on supported Linux/WSL.
- Cost-model/autotuner quality claim is measured-data replay over calibrated shapes. A live CUDA smoke artifact exists, but exact live winner identity is noisy for these tiny kernels.

## Risky / Unsupported Claims
- General PyTorch support is not established.
- Transformer support is not established.
- Physical board execution is not established.

## Code Evidence
- `firmware/host/graph_passes.py`
- `firmware/host/graph_reference_interpreter.py`
- `firmware/host/differential_test_harness.py`
- `firmware/host/pytorch_compiler.py`
- `firmware/host/cuda_autotuner.py`
- `firmware/host/cost_model.py`
- `firmware/host/evaluate_pruned_autotuner.py`
- `firmware/host/isa_simulator.py`
- `firmware/host/run_isa_rtl_bitmatch.py`

## Test Evidence
- `firmware/host/test_graph_passes.py`
- `firmware/host/test_reference_interpreter.py`
- `firmware/host/test_differential_harness.py`
- `firmware/host/test_cuda_autotuner.py`
- `firmware/host/test_isa_simulator.py`

## Validation Commands
- `python -m pytest firmware/host/test_graph_passes.py firmware/host/test_reference_interpreter.py firmware/host/test_differential_harness.py -q`
- `python -m pytest firmware/host/test_graph_passes.py firmware/host/test_cuda_autotuner.py firmware/host/test_differential_harness.py -q`
- `python -m pytest firmware/host/test_differential_harness.py firmware/host/test_isa_simulator.py -q`
- `python firmware/host/run_isa_rtl_bitmatch.py --output-json build/reports/isa_rtl_bitmatch_report.json --output-md build/reports/isa_rtl_bitmatch_report.md`
- `python -m pytest firmware/host/test_footprint_baseline.py firmware/host/test_compiler_smoke.py firmware/host/test_graph_passes.py firmware/host/test_reference_interpreter.py firmware/host/test_isa_simulator.py firmware/host/test_rtl_sim_artifact.py -v`

## Resume-Safe Wording
- Built a pass-based Graph IR compiler (shape inference, Linear+ReLU fusion, dead-code elimination, liveness memory planning, backend legality) and a differential test harness validating backend outputs against a deterministic NumPy interpreter on fixed MLP shapes.
- Verified bit-accurate agreement between a Python uTPU ISA simulator and Verilog RTL on compiled fused MLP programs.
