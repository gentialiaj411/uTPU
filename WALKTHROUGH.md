# uTPU Walkthrough

This is the public top-down tour for the repository as it exists on a clean checkout.

## What It Is

`uTPU` is a scoped compiler/runtime project, not a general PyTorch compiler.

The repo lowers selected PyTorch FX graphs into a custom Graph IR, then into:

- CUDA execution for blocked-FC kernels and graph-op fallback paths
- uTPU ISA programs
- Python ISA simulation
- Verilog RTL simulation

Public in-repo evidence is strongest for:

- blocked-FC MLP flows
- bounded transformer-block validation
- ResNet-18 on the CUDA graph path
- Artix-7 post-route timing/util, RTL cycle attribution, and ISA↔RTL bitmatch (simulation + synth — not on-board)

Live claim browser: [https://gentialiaj411.github.io/uTPU/](https://gentialiaj411.github.io/uTPU/).

On-board FPGA execution evidence is not published in this repo yet.

## Pipeline

```text
PyTorch nn.Module
  -> torch.fx trace
  -> custom Graph IR
  -> pass pipeline
       shape inference
       fusion
       dead-code elimination
       memory planning
       backend legality
  -> lowering
       CUDA backend
       uTPU ISA backend
  -> runtime / simulator / RTL testbench
```

The architectural point is retargeting: one IR feeds both the CUDA path and the uTPU ISA/RTL path.

## Supported Scope

- Batch-1 blocked-FC MLP flows
- Tested single-block transformer patterns
- ResNet-18 on the CUDA graph backend
- `torch.compile(..., backend="utpu")` as a bounded backend with eager fallback for unsupported subgraphs

Not claimed:

- general PyTorch compiler coverage
- full GPT/BERT graph coverage
- production `torch.compile` backend support
- on-board FPGA execution in the published repo

## Verification Layers

- Graph/compiler correctness: differential checks against reference oracles
- CUDA path: committed artifacts plus targeted host tests
- uTPU path: ISA simulation plus RTL simulation
- Repro artifacts: committed JSON under [`bench/results/`](bench/results/)

Important boundary:

- “hardware” in this repo means ISA simulation and RTL simulation unless a claim explicitly says otherwise

## Public Evidence Pointers

- Evidence map: [docs/EVIDENCE.md](docs/EVIDENCE.md)
- Reproduction commands: [docs/REPRO.md](docs/REPRO.md)
- Public writeup: [docs/WRITEUP.md](docs/WRITEUP.md)
- Resume claim inventory: [RESUME_CLAIMS.md](RESUME_CLAIMS.md)
- ResNet-18 CUDA artifact: [bench/results/real_model_end_to_end.json](bench/results/real_model_end_to_end.json)
- Hardware arc (examples): [bench/results/design_space_sweep.json](bench/results/design_space_sweep.json), [bench/results/cycle_attribution.json](bench/results/cycle_attribution.json), [bench/results/steady_state_attribution.json](bench/results/steady_state_attribution.json), [bench/results/utpu_cycle_model_heldout.json](bench/results/utpu_cycle_model_heldout.json), [bench/results/latency_determinism_vs_gpu.json](bench/results/latency_determinism_vs_gpu.json)

## Current Caveats

- Locked-clock GPU latency percentages are still host-sensitive on the published WSL2 laptop setup. Keep those numbers under a `[needs-locked-clock-artifact]` caveat until regenerated on a locked-clock host.
- The latest `main` CI run is publicly visible in GitHub Actions and is currently not green; see the README for the exact run id and commit.
- The tracked RTL requant hardware-readiness gap requires an iverilog-backed verification pass before it can be claimed fixed.
