# uTPU Project State (Canonical Handoff)

Last updated: 2026-05-10
Owner: Genti Aliaj

## Purpose
This is the canonical handoff file for LLM agents and human collaborators.
If code, RTL semantics, metrics, or validation status changes, this file must be updated in the same change.

## Current High-Level Status
- ARRAY_SIZE-aware software tiling is implemented (`ARRAY_SIZE=16` path).
- Blocked FC execution is implemented in host/compiler/runtime.
- Retargetable-compiler scaffold is introduced for blocked FC planning:
  - explicit `MemoryScope` (`REGISTER/SHARED/GLOBAL`)
  - `TargetDesc` abstraction
  - target-agnostic blocked-FC schedule builder
- Backend interface now supports `utpu` and `cuda` blocked-FC lowering paths.
- CUDA blocked-FC backend is implemented with NVRTC + CUDA Driver API execution path (when CUDA runtime dependencies are installed).
- Segmented blocked execution is implemented to handle BRAM limits.
- Compressed + fused full-inference program generation is implemented.
- Fused compressed full-inference fits instruction BRAM (1017 words <= 1024).
- RTL simulation for fused compressed flow passes with Icarus (`rtl_sim_passed=true`).
- Physical FPGA/UART board validation is still pending (no board access).

## Current Architecture (Practical)
1. PyTorch-trained quantized model is exported to int4 weights/scales.
2. `ProgramLoader` builds instruction programs:
   - Legacy 2x2 mode (preserved)
   - Blocked mode
   - Segmented blocked mode
   - Compressed mode
   - Fused compressed full-inference mode
3. ISA includes STORE/LOAD/RUN/FETCH/HALT and BSTORE.
4. RUN semantics include blocked-FC accumulation controls:
   - `acc_clear_en`
   - accumulate RUN (`compute=1, quantize=0, relu=0`)
   - finalize RUN (`compute=0, quantize=1, relu optional`)
5. RTL decode/control executes uploaded instruction programs.
6. Testbenches/scripts validate decode, accumulation, fetch correctness.

## Known Ground-Truth Metrics
Source reports:
- `build/reports/block_runtime_metrics.json`
- `build/reports/compressed_block_program_metrics.json`
- `build/reports/fused_full_inference_metrics.json`
- `build/reports/rtl_fused_sim_metrics.json`
- `build/reports/software_metrics.json`
- `build/reports/cuda_blocked_fc_benchmark.json`
- `build/reports/cuda_blocked_fc_benchmark_fc1_like.json`
- `build/reports/cuda_blocked_fc_benchmark_fc2_like.json`
- `build/reports/cuda_blocked_fc_benchmark_stress_64x256.json`
- `build/reports/cuda_blocked_fc_benchmark_summary.json`

### Runtime/Compilation Metrics
- `array_size`: 16
- Legacy host tile calls/inference: 515
- Segmented blocked phases: 5
- Fused full-inference phases: 1
- FC1 blocked words: 2701
- FC1 compressed words: 985
- FC2 blocked words: 217
- FC2 compressed words: 85
- Full compressed separate words: 1070
- Full fused words: 1017 (fits 1024)
- Words saved by fusion vs separate compressed: 53
- CUDA blocked-FC benchmark shape: M=10, N=1, K=9 (30 iters, 5 warmup)
- CUDA kernel avg latency (ms): 0.07402333333175193
- CUDA H2D avg latency (ms): 0.5547866666574919
- CUDA D2H avg latency (ms): 0.08737000002080701
- CUDA transfer overhead (% of E2E): 89.66414402374919
- CUDA E2E avg latency (ms): 0.7161800000100508
- cuBLAS baseline avg latency (ms): 0.03237000000050708
- CUDA kernel speed vs cuBLAS (%): 43.72945467807262
- Latest single-run benchmark refresh (M=10, N=1, K=9): kernel=0.07621000000502438 ms, transfer_overhead=89.40370033865204%, kernel_vs_cuBLAS=54.96216594281213% (see `build/reports/cuda_blocked_fc_benchmark.json`)
- Multi-shape cuBLAS comparison (40 iters, 8 warmup):
  - FC1-like (M=9, N=1, K=196): kernel vs cuBLAS = 104.36312233162461%
  - FC2-like (M=10, N=1, K=9): kernel vs cuBLAS = 50.35210058258962%
  - Stress (M=64, N=1, K=256): kernel vs cuBLAS = 64.14334009767711%

### Accuracy/Correctness Metrics
- PyTorch accuracy (%): 90.0
- Tiled/runtime software accuracy (%): 90.0
- Accuracy delta (%): 0.0
- Legacy vs array_block (100-sample) max abs logit diff: 0.0
- RTL fused sim pass: true
- RTL case1 expected/actual bytes: [17, 245] / [17, 245]
- RTL case2 expected/actual bytes: [117, 119] / [117, 119]
- CUDA backend local test status: pass (`test_cuda_backend.py`)
- CUDA GPU execution status in this workspace: working
- CUDA performance metrics vs cuBLAS: unknown in this workspace
- CUDA kernel launch latency: unknown in this workspace
- CUDA host-device copy overhead percentage: unknown in this workspace

Metric source note: Accuracy/equivalence metrics above are from `build/reports/block_runtime_metrics.json` generated on 2026-05-10 in this workspace run.
CUDA status and unknown-metric measurement steps:
1. Install runtime dependency: `python -m pip install cuda-python` (done in this workspace on 2026-05-10).
2. Ensure CUDA Toolkit NVRTC DLL path is visible to the process (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64` in PATH).
3. Run `python firmware/host/test_cuda_backend.py` on an NVIDIA GPU system with driver installed (now passes in this workspace).
4. Add benchmark script (next planned step) to compare blocked-FC kernel throughput against cuBLAS for fixed FC shapes and collect launch/copy timing breakdown.

## Validation Commands (Current)
Run from repo root:

```powershell
python firmware/host/test_compiler_abstractions.py
python firmware/host/test_cuda_backend.py
python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 30 --warmup 5 --output-json build/reports/cuda_blocked_fc_benchmark.json
python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md
python firmware/host/test_compressed_block_program.py
python firmware/host/test_fused_full_inference_program.py
python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md
```

## Key Files (Current)
- Host/compiler/runtime:
  - `firmware/host/backend_lowering.py`
  - `firmware/host/compiler_abstractions.py`
  - `firmware/host/cuda_blocked_fc_backend.py`
  - `firmware/host/benchmark_cuda_blocked_fc.py`
  - `firmware/host/lowering_blocked_fc_utpu.py`
  - `firmware/host/lowering_types.py`
  - `firmware/host/test_compiler_abstractions.py`
  - `firmware/host/test_cuda_backend.py`
  - `firmware/host/program_loader.py`
  - `firmware/host/isa_encoder.py`
  - `firmware/host/tiled_inference.py`
  - `firmware/host/block_runtime_analysis.py`
  - `firmware/host/test_compressed_block_program.py`
  - `firmware/host/test_fused_full_inference_program.py`
  - `firmware/host/run_rtl_fused_sim.py`
- RTL:
  - `rtl/top/top.sv`
  - `rtl/memory/instr_bram.sv`
  - `rtl/unified_buffer/unified_buffer.sv`
  - `rtl/PEArray/pe_controller.sv`
  - `rtl/tb/tb_fused_compressed_program.sv`
  - `rtl/tb/xpm_memory_sdpram_stub.sv`

## Open Risks / Remaining Blockers
- No physical FPGA/UART validation yet.
- No CUDA runtime dependency installed in this workspace (`cuda-python` missing), so CUDA kernel compile/launch path cannot be validated here.
- CUDA compile/launch now works in this workspace after fixing `cuda-python` import-layout compatibility and exposing CUDA NVRTC DLL path.
- cuBLAS-comparative metric is still blocked by missing CuPy (or equivalent cuBLAS benchmark harness) in this workspace.
- cuBLAS comparative baseline is now available in this workspace via CuPy.
- Quantization-scale fidelity vs full training/deployment path should continue to be audited when changing arithmetic.
- Any ISA/RTL semantic change must preserve legacy 2x2 path unless intentionally deprecated.

## Update Protocol (Required for All Agents)
When changing code/RTL/tests/reports, update this file in the same commit:
1. Update `Last updated` date.
2. Update changed metrics with source report paths.
3. Add/remove validation commands if workflow changed.
4. Update blockers/risk section.
5. Add a short change note below.

## Change Notes
- 2026-05-10: Added canonical state file and synchronized with current validated metrics/reports.
- 2026-05-10: Added blocked-FC retargetable scaffold in `firmware/host/compiler_abstractions.py` with explicit memory scopes and target descriptors; `ProgramLoader.build_fc_layer_block_program()` now consumes the shared schedule builder without changing emitted ISA semantics.
- 2026-05-10: Re-ran `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md`; observed equivalence metrics 90.0% vs 90.0% and max abs logit diff 0.0.
- 2026-05-10: Post-change validation run results:
  - `python firmware/host/test_compressed_block_program.py` completed successfully; JSON output reports `decode_semantics_match=true`, FC1 compressed words=985, FC2 compressed words=85.
  - `python firmware/host/test_fused_full_inference_program.py` completed successfully; JSON output reports fused words=1017 (`fits_1024=true`), words saved vs separate compressed=53.
  - `python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md` completed successfully with `rtl_sim_passed=true`.
- 2026-05-10: Added `firmware/host/test_compiler_abstractions.py` to validate new retargetable scaffold behavior:
  - uTPU blocked schedule for FC(9x196) at ARRAY_SIZE=16
  - target mismatch failure path
  - CUDA target descriptor shape
  - command `python firmware/host/test_compiler_abstractions.py` output: `test_compiler_abstractions: PASS`.
- 2026-05-10: Extracted blocked-FC uTPU lowering into `firmware/host/lowering_blocked_fc_utpu.py` and routed `ProgramLoader.build_fc_layer_block_program()` through this lowerer (refactor-only change; no intended semantic differences).
- 2026-05-10: Post-refactor validation run results (all successful):
  - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
  - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> unchanged key metrics: legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0, FC1 words 2701, FC2 words 217.
  - `python firmware/host/test_compressed_block_program.py` -> `decode_semantics_match=true`, FC1 compressed words 985, FC2 compressed words 85.
  - `python firmware/host/test_fused_full_inference_program.py` -> fused words 1017, `fits_1024=true`, words_saved 53.
- 2026-05-10: Added backend interface layer in `firmware/host/backend_lowering.py`:
  - `BlockedFCLoweringRequest` request object
  - `BackendLowerer` protocol
  - `UTPUBackendLowerer` implementation routing to current uTPU blocked lowerer
  - `ProgramLoader` now calls backend interface for blocked FC lowering.
- 2026-05-10: Post-backend-interface validation run results (all successful, unchanged metrics):
  - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
  - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0.
  - `python firmware/host/test_compressed_block_program.py` -> `decode_semantics_match=true`.
  - `python firmware/host/test_fused_full_inference_program.py` -> fused words 1017 (`fits_1024=true`), words_saved 53.
- 2026-05-10: Implemented CUDA blocked-FC backend path:
  - Added `firmware/host/cuda_blocked_fc_backend.py` with:
    - schedule-aware CUDA lowering metadata
    - NVRTC PTX compile + CUDA Driver API launch path
    - deterministic NumPy reference fallback and parity comparison
  - Added shared request type in `firmware/host/lowering_types.py` (removed circular imports).
  - Updated `firmware/host/backend_lowering.py` with backend factory (`utpu`, `cuda`).
  - Updated `firmware/host/program_loader.py` to select backend and execute CUDA blocked-FC path when `backend='cuda'`.
  - Added `firmware/host/test_cuda_backend.py` (passes in this workspace; verifies lowering + fallback execution behavior).
- 2026-05-10: Validation after CUDA integration:
  - `python firmware/host/test_cuda_backend.py` -> `test_cuda_backend: PASS`
  - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
  - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> unchanged key metrics (legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0; FC1 words 2701; FC2 words 217).
  - `python firmware/host/test_compressed_block_program.py` -> `decode_semantics_match=true`.
  - `python firmware/host/test_fused_full_inference_program.py` -> fused words 1017, `fits_1024=true`, words saved 53.
  - `python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md` -> `rtl_sim_passed=true`.
  - Inline smoke check: `ProgramLoader(backend='cuda').execute_fc_layer_blocked(...)` returns structured blocked status in this environment (`executed=False`, reason present) due to missing CUDA runtime deps.
- 2026-05-10: Installed CUDA Python dependency in workspace:
  - `python -m pip install --upgrade pip`
  - `python -m pip install cuda-python`
  - Post-install detection: `CUDAEnvironmentStatus(cuda_python_available=True, runtime_available=False, reason='CUDA runtime/NVRTC unavailable: ... nvrtc*.dll ...')`.
  - Updated CUDA backend detection to support both `cuda` and `cuda.bindings` import layouts and to fail gracefully with explicit blocked reason if NVRTC DLLs are missing.
  - Post-fix tests:
    - `python firmware/host/test_cuda_backend.py` -> `test_cuda_backend: PASS`
    - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
    - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> unchanged metrics (legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0).
- 2026-05-10: Resolved CUDA runtime path issue and validated real execution:
  - Confirmed NVIDIA driver/GPU availability via `nvidia-smi` (Driver 596.21, CUDA 13.2, RTX 5070 Laptop GPU).
  - Located NVRTC DLLs at:
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\nvrtc64_130_0.dll`
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\nvrtc-builtins64_132.dll`
  - Added CUDA bin path to User PATH and current session PATH.
  - Fixed CUDA Driver API call signature compatibility (`cuCtxCreate(None, 0, dev)`) for current `cuda.bindings` API.
  - Verified environment/runtime:
    - `detect_cuda_environment()` -> `cuda_python_available=True`, `runtime_available=True`
  - Verified CUDA execution path:
    - `CUDABlockedFCExecutor.execute(...)` -> `executed=True`, `bit_exact_match_vs_numpy_reference=True`, `max_abs_diff_vs_numpy_reference=0`.
  - Regression after CUDA fixes:
    - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
    - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> unchanged key metrics (legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0).
- 2026-05-10: Added CUDA benchmark script `firmware/host/benchmark_cuda_blocked_fc.py` and generated report `build/reports/cuda_blocked_fc_benchmark.json`.
  - Command:
    - `python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 30 --warmup 5 --output-json build/reports/cuda_blocked_fc_benchmark.json`
  - Measured results:
    - kernel_avg_ms = 0.07084666666704227
    - h2d_avg_ms = 0.5424600000007255
    - d2h_avg_ms = 0.09284000000396493
    - transfer_overhead_pct_of_e2e = 89.96714563548639
    - end_to_end_avg_ms = 0.7061466666717328
  - cuBLAS baseline status:
    - initially unavailable (`cupy` not installed), so `% of cuBLAS throughput` remained unknown pending baseline setup.
- 2026-05-10: Installed CuPy and re-ran CUDA benchmark:
  - Install command:
    - `python -m pip install cupy-cuda12x`
  - Initial rerun failed in baseline phase due to cuBLAS backend load error:
    - `ImportError: DLL load failed while importing cublas: The specified module could not be found.`
  - Updated `firmware/host/benchmark_cuda_blocked_fc.py` to treat cuBLAS baseline failures as non-fatal and continue reporting kernel/transfer metrics.
  - Rerun command:
    - `python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 30 --warmup 5 --output-json build/reports/cuda_blocked_fc_benchmark.json`
  - Latest measured results:
    - kernel_avg_ms = 0.05775666666067991
    - h2d_avg_ms = 0.46537000000625994
    - d2h_avg_ms = 0.0732899999889014
    - transfer_overhead_pct_of_e2e = 90.31605421348695
    - end_to_end_avg_ms = 0.5964166666558413
  - Intermediate issue observed and resolved:
    - `cupy-cuda13x` initially appeared installed but `cupy` module was not importable; forced reinstall fixed module availability.
    - command used: `python -m pip install --force-reinstall --no-cache-dir cupy-cuda13x==14.0.1`
  - Final rerun after repair:
    - `python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 30 --warmup 5 --output-json build/reports/cuda_blocked_fc_benchmark.json`
    - kernel_avg_ms = 0.07402333333175193
    - h2d_avg_ms = 0.5547866666574919
    - d2h_avg_ms = 0.08737000002080701
    - transfer_overhead_pct_of_e2e = 89.66414402374919
    - end_to_end_avg_ms = 0.7161800000100508
    - cuBLAS avg_ms = 0.03237000000050708
    - kernel_speed_vs_cublas_pct = 43.72945467807262
  - Post-fix CUDA regression:
    - `python firmware/host/test_cuda_backend.py` -> `test_cuda_backend: PASS`.
- 2026-05-10: Ran multi-shape CUDA benchmark sweep and wrote summary report:
  - Driver command (Python subprocess sweep):
    - runs `firmware/host/benchmark_cuda_blocked_fc.py` for shapes:
      - FC1-like: M=9, K=196
      - FC2-like: M=10, K=9
      - stress: M=64, K=256
    - each with `--iters 40 --warmup 8`
    - outputs:
      - `build/reports/cuda_blocked_fc_benchmark_fc1_like.json`
      - `build/reports/cuda_blocked_fc_benchmark_fc2_like.json`
      - `build/reports/cuda_blocked_fc_benchmark_stress_64x256.json`
      - `build/reports/cuda_blocked_fc_benchmark_summary.json`
  - Summary metrics:
    - FC1-like: kernel_avg_ms=0.07958750001648696, transfer_overhead_pct=89.59245988434678, kernel_speed_vs_cublas_pct=104.36312233162461
    - FC2-like: kernel_avg_ms=0.07277749999730077, transfer_overhead_pct=90.22126524652363, kernel_speed_vs_cublas_pct=50.35210058258962
    - stress_64x256: kernel_avg_ms=0.09243750000109685, transfer_overhead_pct=88.82103067165454, kernel_speed_vs_cublas_pct=64.14334009767711
  - Regression checks after sweep:
    - `python firmware/host/test_cuda_backend.py` -> `test_cuda_backend: PASS`
    - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
    - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> unchanged key correctness metrics.
- 2026-05-10: Full verification pass completed after CUDA and benchmark integration:
  - `python firmware/host/test_compiler_abstractions.py` -> `test_compiler_abstractions: PASS`
  - `python firmware/host/test_cuda_backend.py` -> `test_cuda_backend: PASS`
  - `python firmware/host/test_compressed_block_program.py` -> `decode_semantics_match=true`, FC1 compressed words 985, FC2 compressed words 85
  - `python firmware/host/test_fused_full_inference_program.py` -> fused words 1017, `fits_1024=true`, words_saved 53
  - `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md` -> legacy 90.0%, array_block 90.0%, max_abs_logit_diff 0.0
  - `python firmware/host/benchmark_cuda_blocked_fc.py --m 10 --k 9 --iters 30 --warmup 5 --output-json build/reports/cuda_blocked_fc_benchmark.json` -> kernel_avg_ms 0.07621000000502438, transfer_overhead_pct_of_e2e 89.40370033865204, kernel_speed_vs_cublas_pct 54.96216594281213
  - `python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md` -> `rtl_sim_passed=true`
