# Evidence Map

This page links the public claims in [`README.md`](../README.md) to tracked code, tests, and artifacts.

## Differential Testing

- Claim: the supported compiler/runtime paths are checked against a reference oracle instead of being asserted from implementation intent alone.
- Code: [`firmware/host/differential_test_harness.py`](../firmware/host/differential_test_harness.py)
- Test: `python -m pytest firmware/host/test_differential_harness.py -q`
- Notes:
  - the CUDA oracle path is platform-dependent
  - the uTPU backend in this harness is software emulation, not board execution

## Cost Model And Pruned Autotuner

- Claim: blocked-FC autotuning uses a cost model to prune candidate schedules while keeping replay quality bounded.
- Code:
  - [`firmware/host/cost_model.py`](../firmware/host/cost_model.py)
  - [`firmware/host/cuda_autotuner.py`](../firmware/host/cuda_autotuner.py)
- Artifacts:
  - [`bench/results/cost_model_heldout.json`](../bench/results/cost_model_heldout.json)
  - [`bench/results/selection_ab.json`](../bench/results/selection_ab.json)
- Tests:
  - `python -m pytest firmware/host/test_cost_model_heldout.py -q`
  - `python -m pytest firmware/host/test_selection_ab.py -q`
- Caveat:
  - runtime-consumption wiring is a stable public claim
  - wall-clock percentage deltas on the unlocked-clock WSL2 laptop host remain `[needs-locked-clock-artifact]`

## Python ISA Simulator / RTL Bitmatch

- Claim: the Python ISA simulator and Verilog RTL produce bit-identical outputs for the covered fused blocked-FC programs.
- Code:
  - [`firmware/host/isa_simulator.py`](../firmware/host/isa_simulator.py)
  - [`firmware/host/run_isa_rtl_bitmatch.py`](../firmware/host/run_isa_rtl_bitmatch.py)
  - [`rtl/tb/tb_fused_compressed_program.sv`](../rtl/tb/tb_fused_compressed_program.sv)
- Tests:
  - `python -m pytest firmware/host/test_isa_simulator.py -q`
  - `python -m pytest firmware/host/test_rtl_sim_artifact.py -q`
- Caveat:
  - this is simulation evidence
  - it is not an on-board FPGA execution claim

## ResNet-18 CUDA Graph Path

- Claim: ResNet-18 lowers end-to-end through FX -> Graph IR -> graph-op execution on the CUDA path, with committed parity evidence.
- Artifact: [`bench/results/real_model_end_to_end.json`](../bench/results/real_model_end_to_end.json)
- Code:
  - [`firmware/host/run_real_model_end_to_end.py`](../firmware/host/run_real_model_end_to_end.py)
  - [`firmware/host/test_real_model_end_to_end.py`](../firmware/host/test_real_model_end_to_end.py)
- Caveat:
  - check `execution_backend` in the artifact
  - this is a CUDA-path claim, not a uTPU full-model execution claim

## Board-Fit Audit

- Claim: instruction BRAM fit is audited explicitly for reference RTL configurations, instead of being inferred.
- Artifact: [`bench/results/board_fit_audit.json`](../bench/results/board_fit_audit.json)
- Code:
  - [`firmware/host/board_config.py`](../firmware/host/board_config.py)
  - [`firmware/host/run_board_fit_audit.py`](../firmware/host/run_board_fit_audit.py)
  - [`firmware/host/test_board_fit_audit.py`](../firmware/host/test_board_fit_audit.py)
- Caveat:
  - board-fit evidence is not the same as board-execution evidence

## Scheduler RTL Cross-Check

- Claim: the scheduler’s percentage cycle reduction survives contact with the RTL FSM, and fetch-byte equality still holds for the checked case.
- Artifact: [`bench/results/scheduler_rtl_crosscheck.json`](../bench/results/scheduler_rtl_crosscheck.json)
- Code:
  - [`firmware/host/run_scheduler_rtl_crosscheck.py`](../firmware/host/run_scheduler_rtl_crosscheck.py)
  - [`firmware/host/test_scheduler_rtl_crosscheck.py`](../firmware/host/test_scheduler_rtl_crosscheck.py)
  - [`rtl/tb/tb_scheduler_cycles.sv`](../rtl/tb/tb_scheduler_cycles.sv)
- Caveat:
  - this is RTL simulation evidence, not on-board timing evidence

## Public Boundaries

- On-board FPGA execution evidence is still out of scope for the published repo state.
- Any latency or gap percentage derived from unlocked-clock GPU timing should stay under `[needs-locked-clock-artifact]` until regenerated on a locked-clock host.
