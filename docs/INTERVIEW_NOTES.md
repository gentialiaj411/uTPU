# Interview Deep-Dive Notes

## Whiteboard Pipeline

```text
Model tensors (int4 weights/acts)
        |
        v
BlockedFCLoweringRequest
(firmware/host/lowering_types.py)
        |
        v
BlockedFCProblem + build_blocked_fc_schedule
(firmware/host/compiler_abstractions.py)
        |
        v
create_backend_lowerer(name)
(firmware/host/backend_lowering.py)
        |
        +------------------------------------+
        |                                    |
        v                                    v
uTPU lowerer                          CUDA lowerer/executor
lowering_blocked_fc_utpu.py           cuda_blocked_fc_backend.py
        |                                    |
        v                                    v
ISAEncoder program bytes              NVRTC compile -> PTX -> launch
(isa_encoder.py)                      (CUDA driver API)
        |                                    |
        v                                    v
ProgramLoader runtime                 Timed execute + parity/metrics
(upload/start/fetch over UART)        (kernel/h2d/d2h/e2e)
```

## 15 Hard Questions With Concrete Answers

1. How is retargetability actually enforced rather than claimed?
- Shared request and schedule are centralized in `firmware/host/lowering_types.py` and `firmware/host/compiler_abstractions.py`; backend divergence is explicit in `firmware/host/backend_lowering.py`.

2. Where does backend-specific logic begin?
- At `create_backend_lowerer(name)` in `firmware/host/backend_lowering.py`. uTPU and CUDA lowerers are split into separate files.

3. What prevents silent schedule/backend mismatch?
- `build_blocked_fc_schedule` checks `target.array_size == problem.array_size` and raises if mismatched (`firmware/host/compiler_abstractions.py`).

4. How is uTPU blocked lowering represented?
- `lower_blocked_fc_program_utpu(...)` pads tensors to block boundaries, emits STORE/LOAD/RUN/FETCH/HALT patterns via `ISAEncoder`, and returns metadata including BRAM fit (`firmware/host/lowering_blocked_fc_utpu.py`).

5. How is CUDA lowering/execution represented?
- `CUDABackendLowerer`/`CUDABlockedFCExecutor` in `firmware/host/cuda_blocked_fc_backend.py` compile kernel source with NVRTC, launch via CUDA driver API, and report timing/parity metrics.

6. What is the strongest correctness evidence?
- Five locked runs show legacy vs array-block max abs logit diff = 0.0 and array-block accuracy = 90.0% on 100 samples (`benchmarks/summary.json`, per-run `benchmarks/run_*/block_runtime_metrics.json`).

7. How do you know fused inference program actually fits BRAM?
- `test_fused_full_inference_program.py` reports `new_fused_full_inference_words=1017` and `fits_1024=true`; stable across all five locked runs.

8. How do you validate RTL behavior without board access?
- `run_rtl_fused_sim.py` compiles/runs `rtl/tb/tb_fused_compressed_program.sv` under Icarus and records `rtl_sim_passed` + `total_cycles` in JSON.

9. Why does the project still include legacy 2x2 path?
- Legacy path serves compatibility and regression comparison while blocked path matures (`ProgramLoader.execute2x2MatMul` and block-runtime comparison flow).

10. Where is performance currently limited on GPU?
- Transfer overhead dominates end-to-end time in locked runs (median ~89% across tested shapes), indicating kernel speed alone is not the primary bottleneck.

11. How was reproducibility made interview-defensible?
- One command `make bench` runs five passes for each category and stores raw JSON under `benchmarks/run_01..run_05` with min/median/max summary in `benchmarks/summary.json`.

12. Are there abstraction leaks remaining?
- Yes: `ProgramLoader.execute_fc_layer_blocked` contains a direct `"cuda"` runtime branch (`firmware/host/program_loader.py:551`). Documented in `docs/ABSTRACTION_AUDIT.md`.

13. Why is backend factory string switching acceptable?
- `backend_lowering.py` is the intended single divergence seam; concentrated string-switching here is preferable to distributed checks.

14. What evidence exists that uTPU and CUDA share more than a name?
- Both consume the same `BlockedFCLoweringRequest` shape and derive block structure from shared schedule modeling before lowering diverges.

15. What would you improve first for production quality?
- Introduce backend runtime protocol to remove `ProgramLoader` backend branch; add autotuning and stricter CUDA benchmark environment controls.

## Honest Weaknesses To Discuss

- Backend runtime dispatch is not fully abstracted yet (`program_loader.py` still branches on backend string).
- Operator coverage is narrow (MLP blocked FC focus); abstraction generality beyond this domain remains to be proven.
- CUDA small-shape benchmark variability is non-trivial; locked min/median/max show spread, especially in cuBLAS-relative percentages.
- No physical board validation in this pass; claims are software + RTL simulation scoped.
