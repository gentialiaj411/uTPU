# Reproducibility Guide

This guide lists the public commands that regenerate or validate the repo’s published evidence.

## Baseline Host Regression

```bash
make test-host
```

This is the canonical host regression command and mirrors the CI test list in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Focused Public Checks

```bash
python -m pytest firmware/host/test_differential_harness.py -q
python -m pytest firmware/host/test_isa_simulator.py -q
python -m pytest firmware/host/test_graph_passes.py -q
python -m pytest firmware/host/test_real_model_end_to_end.py -q
python -m pytest firmware/host/test_cost_model_heldout.py -q
python -m pytest firmware/host/test_selection_ab.py -q
```

## Demo / Inspection

```bash
python examples/inspect_compiler_pipeline.py
python examples/visualize_compiler_pipeline.py
```

## RTL-Dependent Checks

These require `iverilog` to be installed and available to the repo tooling.

```bash
make sim-iverilog-fused
make sim-iverilog-batched
make sim-iverilog-scheduler-cross-check
make sim-iverilog-latency
```

If `iverilog` is unavailable, do not promote RTL-fix claims that depend on those runs.

## Public Artifact Map

- ResNet-18 CUDA graph path:
  - artifact: [`bench/results/real_model_end_to_end.json`](../bench/results/real_model_end_to_end.json)
  - command: `python firmware/host/run_real_model_end_to_end.py --output bench/results/real_model_end_to_end.json`
- Held-out cost model:
  - artifact: [`bench/results/cost_model_heldout.json`](../bench/results/cost_model_heldout.json)
  - command: `python firmware/host/run_cost_model_heldout.py`
- Runtime schedule-consumption A/B:
  - artifact: [`bench/results/selection_ab.json`](../bench/results/selection_ab.json)
  - command: `python firmware/host/run_selection_ab.py`
- Board-fit audit:
  - artifact: [`bench/results/board_fit_audit.json`](../bench/results/board_fit_audit.json)
  - command: `python firmware/host/run_board_fit_audit.py`
- Scheduler RTL cross-check:
  - artifact: [`bench/results/scheduler_rtl_crosscheck.json`](../bench/results/scheduler_rtl_crosscheck.json)
  - command: `python firmware/host/run_scheduler_rtl_crosscheck.py`

## Public Caveats

- The repo’s public hardware evidence is simulation-based unless a page explicitly says otherwise.
- GPU timing percentages on the published unlocked-clock WSL2 laptop setup remain `[needs-locked-clock-artifact]`.
- Latest `main` CI status should be read from the badge plus the public Actions run history in [`README.md`](../README.md).
