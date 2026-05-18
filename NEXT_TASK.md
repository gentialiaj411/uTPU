# NEXT_TASK.md

## Current Priority
Schedule-aware pruning follow-up:
1. Pairwise ordering objective is implemented in `calibrate_cost_model.py` and verified with no-new-CUDA refit.
   - current objective uses schedule-median pairs touching the winner/near-winner set and skips measured pairs within `1%` of best.
2. Latest replay (`python firmware/host/evaluate_pruned_autotuner.py --top-k 4`):
   - policy containment `0.8333` vs strict `0.7917`
   - policy max regression `0.49%` vs strict `5.90%`
   - policy within-1% fraction `1.0000`
   - policy avg profiled candidates `4.92`
   - policy search reduction `3.25x`
3. Caveat:
   - aggregate latency fit is below the original 5-term artifact: log_R2 `0.9204`, MAPE `12.85%`, p95 `35.09%`
   - calibration ranking top1 remains weak; this is still a pruning-safety model, not a single-winner predictor.
4. Next cost-model focus:
   - live smoke comparison has been run on four shapes (`build/reports/live_autotuner_comparison.json`): policy top-4 elapsed `1.58s` vs exhaustive `3.50s`
   - if claiming live quality, improve the live timing methodology for tiny kernels first; current quality claim should remain measured-data replay
   - consider reporting near-winner containment because exact winner identity is noisy when schedule latencies differ by less than `1%`
   - keep strict-vs-policy replay reporting as-is; avoid collapsing these metrics in summaries.

#7-#9 status:
- Differential harness now includes a `torch.compile`/TorchInductor oracle entry.
  - Local Windows run marks TorchInductor as skipped with `WinError 50`; do not claim TorchInductor pass coverage until rerun on a supported Linux/WSL stack.
  - CUDA compiled backend and uTPU quantized emulation still pass harness tolerance.
- Python ISA simulator now exists:
  - `firmware/host/isa_simulator.py`
  - `firmware/host/test_isa_simulator.py`
  - `firmware/host/run_isa_rtl_bitmatch.py`
  - `build/reports/isa_rtl_bitmatch_report.json` shows `all_isa_expected_bitmatch=true` and `all_isa_rtl_bitmatch=true`
  - validated cases: `case1_single_k` bytes `[17,245]`, `case2_multi_k` bytes `[117,119]`
- README front page was rewritten around scoped claims and evidence links.
  - Evidence summary: `docs/EVIDENCE.md`
  - Terminal preview SVG: `docs/inspect_compiler_pipeline_demo.svg`

Cost model status:
- CTA-working-set analytical form implemented for CUDA blocked-FC latency modeling.
- Fresh CUDA timing confirms the model form:
  - `build/reports/cost_model_calibration.json`
  - command: `python firmware/host/calibrate_cost_model.py --warmup 8 --iters 20 --repeat-launches-per-sample 32`
  - log_R2 `0.9323`, MAPE `10.68%`, p95 abs relative error `25.41%`
- Holdout validation artifact exists:
  - `build/reports/cost_model_holdout_validation.json`
  - requested shape-triplet split test metrics: log_R2 `0.9283`, MAPE `10.11%`, p95 `23.42%`
  - stricter actual-layer-shape split test metrics: log_R2 `0.6644`, MAPE `12.53%`, p95 `37.85%`
- Cost-model-pruned autotuner evidence exists:
  - `build/reports/pruned_autotuner_report.json`
  - top-4 of 16 candidates gives `4.00x` search reduction
  - calibrated-shape measured-data replay: mean quality regression `0.71%`, p95 `3.58%`, max `4.41%`

Next cost-model focus:
1. if desired, run the pruned autotuner live on a smaller subset to confirm wall-clock tuning reduction beyond measured-data replay
2. if claiming broader unseen-shape prediction, audit strict held-out `(in=64,out=512,tpb=256)` residuals before final wording
3. keep resume wording scoped to measured/calibrated shape-grid prediction and pruning

Deliverable B status:
- Completed and re-validated on 2026-05-17 via `python -m pytest firmware/host/test_reference_interpreter.py -q` (`2 passed`).
Deliverable C status:
- Completed and re-validated on 2026-05-18 via `python -m pytest firmware/host/test_differential_harness.py -q` (`1 passed`).
- `build/reports/differential_test_report.json` refreshed in-session.
- Fixture now uses seeded signed sparse weights/inputs with tiny deterministic jitter; report max_abs_error is nonzero but within `1e-5`.
- Memory planning pass added and validated:
  - `memory_planning` now runs before backend legality
  - `build/reports/memory_plan_report.json` shows `33.33%` activation allocation reduction on the liveness sample

Next differential-harness focus:
 Harden Deliverable C evidence:
1. rerun TorchInductor oracle on Linux/WSL where `torch.compile(..., backend="inductor")` works before claiming that oracle as passing
2. add a schema-lock test for `build/reports/differential_test_report.json`
3. run the differential harness once on a CUDA-unavailable machine to validate skip behavior
4. optionally add hardware-in-the-loop uTPU differential check when board runtime is available

## Strict Scope
- Keep blocked-FC subset scope unchanged.
- Do not claim board execution unless verified by artifact.
