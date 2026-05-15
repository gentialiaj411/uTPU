# uTPU Project State

Last updated: 2026-05-15

## Current Scope

This project is currently a blocked-FC compiler/runtime with uTPU and CUDA lowering paths. The core execution path remains the existing int4 blocked fully connected scheduler and backend lowerers.

This is not yet a full general-purpose model compiler. Transformer support, arbitrary PyTorch operator support, and `torch.compile` integration are not implemented in this state.

## Key Host Compiler Files

- `firmware/host/compiler_abstractions.py` - target descriptors, blocked-FC problem model, and schedule construction.
- `firmware/host/lowering_types.py` - shared `BlockedFCLoweringRequest` schema.
- `firmware/host/backend_lowering.py` - backend lowerer protocol and uTPU/CUDA lowerer factory.
- `firmware/host/lowering_blocked_fc_utpu.py` - uTPU blocked-FC ISA lowering.
- `firmware/host/cuda_blocked_fc_backend.py` - CUDA blocked-FC lowering metadata and execution/reference path.
- `firmware/host/program_loader.py` - host program construction, upload, and blocked-FC runtime helpers.

## Graph Frontend Milestone

Milestone 1 adds an initial graph-level compiler frontend for simple MLP-style FX graphs while reusing the existing blocked-FC backend infrastructure.

New files:

- `firmware/host/graph_ir.py` - minimal tensor graph IR with `GraphIR`, `TensorValue`, `OpNode`, op kind constants, shape/dtype metadata, and producer/consumer links.
- `firmware/host/fx_importer.py` - PyTorch FX importer for simple `nn.Linear`, `torch.relu` / `nn.ReLU`, easy add nodes, and trivial view-like methods.
- `firmware/host/graph_lowering.py` - graph planning layer that identifies Linear/ReLU patterns and creates `BlockedFCLoweringRequest` objects for blocked-FC lowering.
- `firmware/host/pytorch_compiler.py` - user-facing PyTorch compiler entrypoint that traces a model with FX, imports Graph IR, builds a blocked-FC compile plan, and routes supported Linear ops through the uTPU/CUDA backend lowerers.
- `firmware/host/graph_runtime_plan.py` - graph-level runtime buffer/op plan for supported MLP execution, including inputs, weights, biases, intermediates, outputs, and execution order.
- `firmware/host/compiled_runtime.py` - callable compiled MLP runtime for CUDA-targeted Linear/ReLU/Linear graphs. Supported Linear ops execute through a GPU-resident NVRTC CUDA blocked-FC graph path in `mode="compiled"`; supported bias/ReLU post-ops execute through an NVRTC CUDA elementwise kernel; `mode="reference"` is the explicit PyTorch comparison path. The runtime also exposes a software quantized int4 graph reference for benchmark correctness reporting.
- `firmware/host/cuda_blocked_fc_backend.py` - CUDA blocked-FC executor now separates NVRTC compile/context/module/buffer setup timing from H2D, kernel, and D2H timing, caches CUDA context, compiled modules/functions, and reusable device buffers, includes NVRTC int4 elementwise and quantized blocked-FC kernels for GPU-resident graph execution, restores the executor CUDA context before cached launches, and accepts explicit CUDA blocked-FC schedule parameters.
- `firmware/host/cuda_autotuner.py` - small CUDA blocked-FC schedule tuner with a typed search space, repeated timing, correctness checks against the NumPy reference, and JSON cache/report persistence keyed by M/N/K, dtype mode, array size, and CUDA backend.
- `firmware/host/test_fx_importer.py` - FX import tests for simple MLPs, ReLU, add, and clear unsupported-op failures.
- `firmware/host/test_graph_lowering.py` - graph planning tests for Linear/ReLU routing into blocked-FC request structures.
- `firmware/host/test_pytorch_compiler.py` - end-to-end PyTorch model compiler entrypoint tests for 2-layer MLPs, unsupported models, CUDA target routing, and uTPU instruction-program emission.
- `firmware/host/test_compiled_mlp_execution.py` - callable compiled-runtime tests for 2-layer MLP execution, ReLU behavior, unsupported failures, and uTPU instruction-plan emission.
- `firmware/host/test_cuda_autotuner.py` - tuner search-space, one-shape tuning, correctness, cache roundtrip, opt-in tuned runtime, and fixed-schedule regression tests.
- `examples/compile_tiny_mlp.py` - small demo showing PyTorch model -> FX graph -> Graph IR -> blocked-FC backend lowering plan.
- `examples/run_compiled_tiny_mlp.py` - small demo compiling and executing a tiny MLP through the CUDA callable runtime, then reporting max absolute error versus PyTorch.
- `examples/run_compiled_tiny_mlp_backend.py` - backend-focused demo that prints CUDA backend Linear op counts, fallback/adapter ops, max absolute error, and per-op latency.
- `examples/benchmark_compiled_tiny_mlp.py` - benchmark demo that reports first-call latency, steady-state latency, compile/setup/transfer/kernel/adapter timing, backend op counts, fallback ops, and max absolute error.
- `examples/benchmark_mlp_baselines.py` - baseline comparison across PyTorch eager, optional `torch.compile`, explicit torch matmul/cuBLAS-style Linear execution, compiled first call, compiled warmed steady state, kernel-only, H2D/D2H, and end-to-end timing; writes `build/reports/mlp_baseline_comparison.json`.
- `examples/autotune_cuda_blocked_fc.py` - searches CUDA blocked-FC schedule parameters and writes the best schedule cache/report to `build/reports/cuda_autotune_results.json`.
- `examples/benchmark_tuned_mlp.py` - compares fixed-schedule and opt-in tuned compiled MLP execution; writes `build/reports/tuned_mlp_benchmark.json`.

Important limitation: this frontend imports, plans, and executes simple MLP graphs on the CUDA-targeted callable runtime. In compiled mode, Linear ops route through the existing int4 NVRTC CUDA blocked-FC backend, and supported bias/ReLU post-ops route through an NVRTC CUDA elementwise kernel. `mode="reference"` uses PyTorch for comparison only. It still does not provide general PyTorch model coverage, graph-wide allocation optimization, quantization calibration, physical uTPU board execution from Graph IR, transformer support, or a production `torch.compile` backend. The user-facing `compile_mlp_model(...)` API is a real PyTorch graph compiler/runtime entrypoint for the currently supported MLP subset.

Latest tiny MLP backend benchmark observed during validation (`examples/benchmark_compiled_tiny_mlp.py`, one run on the local CUDA environment):

- first-call wall time: `185.5469 ms`
- steady-state wall time: `1.3364 ms`
- compile time: `93.5132 ms`
- setup time: `89.3727 ms`
- steady-state H2D time: `0.0392 ms`
- steady-state kernel time across 2 Linear ops + 1 elementwise op: `0.7501 ms`
- steady-state D2H time: `0.0516 ms`
- adapter time: `0.0000 ms`
- backend Linear ops executed: `2`
- backend elementwise ops executed: `1`
- fallback/adapter ops: `[]`
- max absolute error vs PyTorch: `0.00000000`
- transfer structure: GPU-resident graph execution performs one activation H2D copy and one final D2H copy per compiled invocation. The prior per-op compiled path copied intermediate outputs back to host between graph ops.

Latest baseline comparison observed during validation (`examples/benchmark_mlp_baselines.py`, one run on the local CUDA environment):

- `tiny_mlp`: PyTorch eager `0.5959 ms`, torch matmul/cuBLAS-style `0.5457 ms`, compiled steady state `1.4703 ms`, compiled kernel `1.0739 ms`, H2D/D2H counts `1/1`, compiled-vs-quantized-reference error `0.0`, quantized-reference-vs-float-PyTorch error `0.0`, fallback ops `[]`; compiled steady state lost to both measured baselines. `torch.compile` was skipped because Triton was unavailable.
- `fc1_like_small`: PyTorch eager `0.5823 ms`, torch matmul/cuBLAS-style `0.5836 ms`, compiled steady state `1.6335 ms`, compiled kernel `1.2180 ms`, H2D/D2H counts `1/1`, compiled-vs-quantized-reference error `0.0`, quantized-reference-vs-float-PyTorch error `310.0`, fallback ops `[]`; compiled steady state lost to both measured baselines.
- `fc2_like_small`: PyTorch eager `0.5908 ms`, torch matmul/cuBLAS-style `0.5708 ms`, compiled steady state `1.6944 ms`, compiled kernel `1.2937 ms`, H2D/D2H counts `1/1`, compiled-vs-quantized-reference error `0.0`, quantized-reference-vs-float-PyTorch error `599.0`, fallback ops `[]`; compiled steady state lost to both measured baselines.
- `stress_256_256_128`: PyTorch eager `0.5910 ms`, torch matmul/cuBLAS-style `0.5282 ms`, compiled steady state `1.4576 ms`, compiled kernel `0.9319 ms`, H2D/D2H counts `1/1`, compiled-vs-quantized-reference error `0.0`, quantized-reference-vs-float-PyTorch error `1417.0`, fallback ops `[]`; compiled steady state lost to both measured baselines.

The formerly large non-tiny `max_abs_error` values were not CUDA padding/layout failures in this validation run. They came from comparing saturated int4 compiled outputs directly against unconstrained float PyTorch outputs. The reports now split backend correctness (`compiled_vs_quantized_reference_max_error`) from quantization drift (`quantized_reference_vs_float_pytorch_max_error`).

Latest CUDA blocked-FC autotune report observed during validation (`examples/autotune_cuda_blocked_fc.py`, one run on the local CUDA environment):

- Search space: `threads_per_block in [32, 64, 128, 256]`, `unroll_factor in [1, 2, 4, 8]`.
- Best per-op kernel medians found: `tiny_fc1` `0.0251 ms` with `{threads_per_block: 256, unroll_factor: 8}`; `tiny_fc2` `0.4187 ms` with `{threads_per_block: 256, unroll_factor: 2}`; `fc1_like_small_fc1` `0.5046 ms` with `{threads_per_block: 32, unroll_factor: 2}`; `shared_64x128_linear` `0.2777 ms` with `{threads_per_block: 32, unroll_factor: 2}`; `fc2_like_small_fc2` `0.0297 ms` with `{threads_per_block: 256, unroll_factor: 2}`; `stress_linear` `0.0319 ms` with `{threads_per_block: 32, unroll_factor: 8}`.
- Tuner correctness error vs the NumPy blocked-FC reference was `0` for all tuned Linear shapes.
- The opt-in tuned compiled MLP benchmark did not improve steady-state wall time in this run: `tiny_mlp` fixed kernel `1.1455 ms` vs tuned `1.2017 ms`, fixed steady `1.5168 ms` vs tuned `2.5510 ms`; `fc1_like_small` fixed kernel `1.4799 ms` vs tuned `1.1166 ms`, fixed steady `1.9713 ms` vs tuned `2.5305 ms`; `fc2_like_small` fixed kernel `1.1793 ms` vs tuned `1.5524 ms`, fixed steady `1.5999 ms` vs tuned `2.9293 ms`. Fixed schedule remains available and is still the default.

Generated reports:

- `build/reports/mlp_baseline_comparison.json`
- `build/reports/cuda_autotune_results.json`
- `build/reports/tuned_mlp_benchmark.json`

## Validation Commands

Run from the repo root:

- `python examples/benchmark_mlp_baselines.py`
- `python examples/autotune_cuda_blocked_fc.py`
- `python examples/benchmark_tuned_mlp.py`
- `python firmware/host/test_cuda_autotuner.py`
- `python firmware/host/test_fx_importer.py`
- `python firmware/host/test_graph_lowering.py`
- `python firmware/host/test_pytorch_compiler.py`
- `python firmware/host/test_compiled_mlp_execution.py`
- `python examples/compile_tiny_mlp.py`
- `python examples/run_compiled_tiny_mlp.py`
- `python examples/run_compiled_tiny_mlp_backend.py`
- `python examples/benchmark_compiled_tiny_mlp.py`
- `python firmware/host/test_compiler_abstractions.py`
- `python firmware/host/test_cuda_backend.py`
- `python firmware/host/test_compressed_block_program.py`
- `python firmware/host/test_fused_full_inference_program.py`
- `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md`

Note: `test_fx_importer.py` and the FX-specific path in `test_graph_lowering.py` require PyTorch. In environments without PyTorch, those tests report a skip for FX coverage rather than adding PyTorch as a project dependency.

## Recent Change Notes

- 2026-05-15: Added initial Graph IR + FX importer milestone for simple MLPs. The new frontend recognizes Linear/ReLU/add/view-like FX nodes where straightforward, rejects unsupported FX nodes with explicit errors, and plans Linear/ReLU sequences toward existing blocked-FC lowering via `BlockedFCLoweringRequest`.
- 2026-05-15: Added `compile_mlp_model(...)` PyTorch compiler entrypoint plus tiny MLP demo. The entrypoint traces PyTorch models with FX, imports Graph IR, reports unsupported ops, and lowers supported Linear/ReLU MLP subgraphs through the existing CUDA/uTPU blocked-FC backend lowerers.
- 2026-05-15: Added callable compiled MLP runtime for CUDA-targeted 2-layer Linear/ReLU/Linear graphs. `compile_mlp_model(...)` now returns a callable compiled object for supported CUDA graphs and continues to emit uTPU instruction plans where board execution is unavailable.
- 2026-05-15: Replaced the CUDA compiled path's PyTorch matmul adapter with real NVRTC CUDA blocked-FC backend execution for supported Linear ops. PyTorch execution remains available only through explicit reference/fallback modes, while ReLU and bias are reported as temporary adapters rather than claimed as generated CUDA.
- 2026-05-15: Added CUDA backend caching and benchmark reporting. The runtime now separates one-time compile/setup costs from steady-state execution, caches CUDA context/modules/functions/device buffers, and exposes `compiled.benchmark(...)` for first-call vs warmed latency reporting.
- 2026-05-15: Removed adapter fallback from the supported Linear/ReLU/Linear compiled path by adding a cached NVRTC CUDA elementwise int4 kernel for bias/ReLU. Supported MLP benchmark now reports `fallback_ops=[]` and `adapter_time_ms=0.0000`.
- 2026-05-15: Added baseline benchmarking and a small CUDA blocked-FC autotuner. Baselines currently show the compiled runtime losing to PyTorch eager and torch matmul/cuBLAS-style execution for the measured tiny/small batch-1 MLPs. The tuner finds faster isolated Linear kernel schedules for several shapes, persists them by shape/dtype/backend, and can be enabled in the compiled runtime, but the latest full MLP tuned benchmark regressed kernel and steady-state time versus fixed schedule.
- 2026-05-15: Split benchmark correctness into compiled-vs-quantized-reference and quantized-reference-vs-float-PyTorch errors. The large non-tiny errors are quantization/saturation drift relative to float PyTorch, while compiled CUDA output is bit-exact against the software int4 graph reference for the measured shapes. The compiled CUDA runtime now uses a GPU-resident graph path with one activation H2D transfer and one final D2H transfer per invocation, avoiding host round trips between Linear/ReLU/Linear ops. Current resident kernels reduce transfer count but do not yet beat PyTorch/cuBLAS end-to-end.
