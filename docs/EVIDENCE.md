# Evidence Map

This page records the current recruiter-facing claims and the local artifacts or commands that regenerate them.

## Differential Testing

- Claim: backend outputs are differentially tested against a NumPy Graph IR reference, TorchInductor when available, CUDA compiled runtime, and uTPU software emulation.
- Code: `firmware/host/differential_test_harness.py`
- Test: `python -m pytest firmware/host/test_differential_harness.py -q`
- Local artifact: `build/reports/differential_test_report.json`
- Current local run: CUDA and uTPU emulation pass on three MLP shapes; `torch.compile_inductor` is present as an oracle but skipped on this Windows stack with `WinError 50`.

## Cost Model And Pruned Autotuner

- Claim: CUDA blocked-FC autotuning uses a schedule-aware analytical cost model plus conservative pruning policy to reduce measured schedule candidates while bounding quality loss on calibrated replay.
- Code: `firmware/host/cost_model.py`, `firmware/host/calibrate_cost_model.py`, `firmware/host/cuda_autotuner.py`, `firmware/host/evaluate_pruned_autotuner.py`
- Calibration artifact: `build/reports/cost_model_calibration.json`
- Current refit mode: `refit_from_existing_measurements=true`; no new CUDA timing was required for the latest coefficient/objective refit.
- Current fit objective: `mean_log_latency_mse_plus_pairwise_ordering`
- Current aggregate fit after filtered pairwise objective: log_R2 `0.9204`, MAPE `12.85%`, p95 absolute relative error `35.09%`.
- Ranking caveat: exact top-1 schedule prediction remains weak; the supported claim is pruning quality, not single-winner prediction.
- Pruned autotuner artifact: `build/reports/pruned_autotuner_report.json`
- Current replay result: policy profiles `4.92` of 16 schedules on average (`3.25x` search reduction), max quality regression `0.49%`, and within-1% fraction `1.00` across 24 calibrated layer shapes.
- Strict top-k comparison in the same artifact: strict top-k max quality regression `5.90%`.
- Live smoke artifact: `build/reports/live_autotuner_comparison.json`
- Live smoke result: exhaustive and policy-pruned tuning both execute on four representative CUDA shapes; policy run completed faster wall-clock on this local run, but tiny kernel timing noise means replay remains the primary quality evidence.

## Python ISA Simulator / RTL Bitmatch

- Claim: Python ISA simulator and Verilog RTL produce bit-identical fetch bytes on compiled fused programs.
- Python simulator: `firmware/host/isa_simulator.py`
- Bitmatch runner: `firmware/host/run_isa_rtl_bitmatch.py`
- RTL testbench: `rtl/tb/tb_fused_compressed_program.sv`
- Test: `python -m pytest firmware/host/test_isa_simulator.py -q`
- Local artifact: `build/reports/isa_rtl_bitmatch_report.json`
- Current local run: `all_isa_expected_bitmatch=True`, `all_isa_rtl_bitmatch=True`
- Cases:
  - `case1_single_k`: expected `[17, 245]`, Python ISA `[17, 245]`, RTL `[17, 245]`
  - `case2_multi_k`: expected `[117, 119]`, Python ISA `[117, 119]`, RTL `[117, 119]`

## Liveness Memory Planning

- Claim: pass-based IR includes liveness-driven activation buffer reuse.
- Code: `firmware/host/graph_passes.py`, `firmware/host/graph_runtime_plan.py`
- Local artifact: `build/reports/memory_plan_report.json`
- Current sample result: naive persistent activation bytes `48`, planned peak bytes `32`, peak reduction `33.33%`.
