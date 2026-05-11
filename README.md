# uTPU Retargetable ML Compiler (uTPU + CUDA)

This project is a retargetable blocked-FC compiler/runtime that drives both a custom uTPU ISA path and a CUDA path from shared compiler structure. The same front-end tensor shapes flow through shared problem modeling, blocked scheduling, and request typing, then diverge in target-specific lowering/codegen and runtime execution. Today the backend dispatch is explicit in both `backend_lowering.py` and one loader-level runtime branch in `ProgramLoader` (`firmware/host/program_loader.py`), so the boundary is mostly centralized but not fully abstracted yet.

## Architecture

```text
Frontend tensors/weights
        |
        v
Blocked FC problem model
(compiler_abstractions.py, lowering_types.py)
        |
        v
Tiling / schedule construction
(build_blocked_fc_schedule)
        |
        v
Backend dispatch (backend_lowering.py)
        |
        +-------------------------------+
        |                               |
        v                               v
uTPU lowering/codegen            CUDA lowering/codegen
(lowering_blocked_fc_utpu.py)    (cuda_blocked_fc_backend.py)
        |                               |
        v                               v
ISA program emission              NVRTC PTX + CUDA launch
(program_loader.py, isa_encoder.py)    (CUDABlockedFCExecutor)
        |                               |
        v                               v
Host runtime / execution          Host runtime / execution
(UART upload/start/fetch)         (timed kernel + transfer path)
```

## What's Actually Retargetable

### Target-agnostic passes and abstractions
- `firmware/host/compiler_abstractions.py`
  - `BlockedFCProblem`
  - `BlockedFCSchedule`
  - `build_blocked_fc_schedule(...)`
  - `MemoryScope` and generic `TargetDesc` fields used for schedule metadata.
- `firmware/host/lowering_types.py`
  - `BlockedFCLoweringRequest` shared request schema for backend lowerers.

### Target-specific logic
- uTPU-specific:
  - `firmware/host/lowering_blocked_fc_utpu.py` (ISA-oriented blocked lowering)
  - `firmware/host/isa_encoder.py` (instruction encoding)
  - `firmware/host/program_loader.py` (UART upload/start/fetch runtime path)
  - `rtl/**` (hardware implementation)
- CUDA-specific:
  - `firmware/host/cuda_blocked_fc_backend.py` (NVRTC PTX generation, CUDA driver launch, timing)

### Controlled divergence point
- `firmware/host/backend_lowering.py`
  - `create_backend_lowerer(name)` is the explicit backend switch.

## Locked Benchmark Numbers (5 runs, min/median/max)

Source: `benchmarks/summary.json` and raw JSON in `benchmarks/run_01..run_05/`.

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| Block runtime correctness: array_block accuracy (%) | 90.0 | 90.0 | 90.0 |
| Block runtime correctness: max abs logit diff | 0.0 | 0.0 | 0.0 |
| CUDA small (M=10,K=9): kernel avg ms | 0.0690 | 0.0828 | 0.0893 |
| CUDA small: transfer overhead % | 87.7588 | 89.2987 | 90.9613 |
| CUDA small: kernel-vs-cuBLAS % | 24.1406 | 34.5432 | 73.4157 |
| CUDA medium (M=9,K=196): kernel avg ms | 0.0776 | 0.0973 | 0.1071 |
| CUDA medium: transfer overhead % | 86.1744 | 88.0981 | 90.0485 |
| CUDA medium: kernel-vs-cuBLAS % | 37.6293 | 41.5117 | 74.5213 |
| CUDA representative-MLP (M=64,K=256): kernel avg ms | 0.0837 | 0.1019 | 0.1375 |
| CUDA representative-MLP: transfer overhead % | 84.8928 | 88.9203 | 94.1685 |
| CUDA representative-MLP: kernel-vs-cuBLAS % | 41.7180 | 86.1927 | 106.2286 |
| Fused inference program size (BRAM words) | 1017 | 1017 | 1017 |
| RTL fused sim pass count (out of 5) | 5 | 5 | 5 |
| RTL fused sim cycle count | 44018 | 44018 | 44018 |

## Reproduce

- Reproducibility prerequisites (exact benchmark environment):
  - Python: `3.14.4`
  - CUDA Toolkit (`nvcc --version`): `13.2` (`V13.2.78`)
  - CuPy: `14.0.1` (`cupy-cuda13x`)
  - Icarus Verilog: `12.0 (devel) (s20150603-1539-g2693dd32b)`
  - GPU: `NVIDIA GeForce RTX 5070 Laptop GPU` (driver `596.21`)
  - CPU: `Intel(R) Core(TM) Ultra 9 275HX`
  - These exact values are also embedded in `benchmarks/summary.json` top-level `provenance` and per-run `runs_metadata`.

- Full lock pass (all results above, 5 runs each):
  - `make bench`
  - Equivalent: `python firmware/host/lock_benchmarks.py --runs 5`
- Block runtime correctness only:
  - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md`
- CUDA blocked FC (single shape run):
  - `python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 40 --warmup 8 --output-json build/reports/cuda_blocked_fc_benchmark.json`
- Fused inference program size:
  - `python firmware/host/test_fused_full_inference_program.py`
- RTL fused simulation (pass/fail + cycle count):
  - `python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md`

## Limitations

- Transfer overhead dominates end-to-end CUDA latency on tested shapes (mid/high-80s to low-90s %).
- No autotuning of tile/schedule parameters.
- Operator scope is MLP-style blocked FC flow; broader operator coverage is not implemented.
- cuBLAS-relative numbers vary run-to-run on tiny workloads; lock file captures variability explicitly.
- Board execution is intentionally out of scope in this artifact pass; this repo state focuses on simulation/software-backed validation.
