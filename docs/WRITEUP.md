# Retargeting a Blocked-FC Compiler Path Across uTPU and CUDA

## Motivation

The project started with a working uTPU-oriented blocked FC flow, but the real architectural question was whether the compiler structure could survive a second backend without duplicating logic. The goal was not “add CUDA code” in isolation; it was to prove that one compiler surface could carry shared tensor/block semantics into two materially different execution targets.

## IR and Compiler Structure Choices

The most important design decision was to make the blocked FC problem explicit and serializable before lowering:

- `firmware/host/lowering_types.py` defines `BlockedFCLoweringRequest` as a backend-neutral request object.
- `firmware/host/compiler_abstractions.py` defines `BlockedFCProblem`, `BlockedFCSchedule`, and `build_blocked_fc_schedule(...)`.
- `MemoryScope` and `TargetDesc` encode capacity/placement metadata without forcing target-specific codegen decisions in the scheduler.

This gives a clear pivot point: problem/schedule are shared, lowering is target-owned.

## What Was Hardest About Adding the Second Backend

The difficult part was not kernel math. It was boundary control:

1. Keeping schedule logic target-agnostic while supporting backend constraints.
2. Preventing runtime orchestration from becoming backend-specific glue code.
3. Preserving correctness/equivalence while introducing a second execution path.

The current architecture resolves (1) well through shared schedule construction, but audit findings show one meaningful leak for (2): `ProgramLoader.execute_fc_layer_blocked` still directly branches on `"cuda"` (`firmware/host/program_loader.py:551`) instead of fully delegating execution via a backend runtime protocol.

## Host Runtime Emission Approach

uTPU path:

- Lowering emits ISA-oriented programs (`firmware/host/lowering_blocked_fc_utpu.py`, `firmware/host/isa_encoder.py`).
- Runtime handles upload/start/fetch over UART (`firmware/host/program_loader.py`).
- Fused compressed inference is generated in one program and validated in RTL simulation.

CUDA path:

- Backend-lowering/execution lives in `firmware/host/cuda_blocked_fc_backend.py`.
- NVRTC compiles kernel source to PTX and launches through CUDA driver APIs.
- Benchmark harness reports kernel time, transfer time breakdown, and cuBLAS-relative baseline (`firmware/host/benchmark_cuda_blocked_fc.py`).

Both paths are driven from the same blocked FC request shape and shared schedule metadata.

## Benchmark Methodology

To make claims reproducible, all benchmark categories are locked with five runs each and raw JSON outputs committed:

- Runner: `python firmware/host/lock_benchmarks.py --runs 5` (also `make bench`).
- Raw artifacts: `benchmarks/run_01..run_05/*.json`.
- Summary: `benchmarks/summary.json` with min/median/max.

Locked highlights (5-run medians):

- Block correctness: 90.0% array-block accuracy, 0.0 max abs logit diff vs legacy.
- CUDA (M=64,K=256): 0.1019 ms kernel avg, 88.9203% transfer overhead, 86.1927% kernel-vs-cuBLAS.
- Fused program size: 1017 BRAM words (stable).
- RTL fused sim: 5/5 pass, 44018 cycles.

## What I Would Do Differently

1. Separate runtime execution behind a backend protocol so `ProgramLoader` never checks backend strings directly.
2. Add deterministic environment pinning for CUDA benchmarking (clock control/affinity/warmup governance) to reduce variance in cuBLAS-relative microbenchmarks.
3. Extend operator coverage beyond blocked FC to pressure-test whether current abstractions generalize or were overfit to MLP-style kernels.
4. Add lightweight autotuning for tile choices and transfer batching; current fixed-shape behavior leaves performance on the table for small shapes where transfer dominates.

## Bottom Line

The strongest outcome is architectural, not just performance: a shared blocked-FC compiler core now supports two backends with reproducible metrics and explicit seams. The remaining boundary leak is known, documented, and tractable.
