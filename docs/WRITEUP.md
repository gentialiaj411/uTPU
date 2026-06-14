# uTPU Public Writeup

## Summary

`uTPU` is a retargetable ML compiler/runtime project with a narrow, explicit scope:

- PyTorch FX import
- custom Graph IR and passes
- CUDA blocked-FC and graph-op execution
- uTPU ISA lowering
- Python ISA simulation
- Verilog RTL simulation

The repo’s strongest public claims are about scoped compiler correctness, explicit artifact-backed measurement, and simulation-backed ISA/RTL agreement.

## What Is Publicly Supported

- blocked-FC MLP flows
- tested transformer-block patterns
- ResNet-18 on the CUDA graph path

The repo does not publicly claim:

- general PyTorch compiler coverage
- full-model uTPU execution for ResNet or transformers
- on-board FPGA execution evidence

## Evidence Highlights

- Differential compiler/runtime checking:
  - [`docs/EVIDENCE.md#differential-testing`](EVIDENCE.md#differential-testing)
- ResNet-18 CUDA graph-path parity artifact:
  - [`bench/results/real_model_end_to_end.json`](../bench/results/real_model_end_to_end.json)
- Held-out cost-model artifact:
  - [`bench/results/cost_model_heldout.json`](../bench/results/cost_model_heldout.json)
- Runtime schedule-consumption artifact:
  - [`bench/results/selection_ab.json`](../bench/results/selection_ab.json)
- ISA/RTL cross-check:
  - [`rtl/tb/tb_fused_compressed_program.sv`](../rtl/tb/tb_fused_compressed_program.sv)
- Board-fit audit:
  - [`bench/results/board_fit_audit.json`](../bench/results/board_fit_audit.json)

## Measurement Discipline

The repo deliberately separates stable evidence from unstable evidence:

- stable: committed artifacts, simulator parity, RTL bitmatch, board-fit audit, structural compiler behavior
- unstable on the published unlocked-clock WSL2 laptop host: GPU latency and gap percentages

Public rule:

- if a claim depends on unlocked-clock GPU timing percentages, keep it under `[needs-locked-clock-artifact]`

That caveat applies to percentage-gap interpretations around:

- cuBLAS / Inductor baseline comparisons
- fused-region latency deltas
- runtime realized-regret percentages

The artifacts remain useful, but those percentages should not be promoted as stable public headlines until regenerated on a locked-clock host.

## Reproduction

Use the command map in [docs/REPRO.md](REPRO.md).

Key public entry points:

```bash
make test-host
python -m pytest firmware/host/test_differential_harness.py -q
python -m pytest firmware/host/test_isa_simulator.py -q
python -m pytest firmware/host/test_real_model_end_to_end.py -q
python examples/inspect_compiler_pipeline.py
```

## Current Public Gaps

- Latest `main` CI is not currently green; see the exact run id in [`README.md`](../README.md).
- The tracked requant RTL readiness gap still requires an iverilog-backed verification pass before it can be closed publicly.
- On-board FPGA execution evidence remains out of scope for the published repo state.
