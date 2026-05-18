# uTPU: Scoped ML Compiler + CUDA/uTPU Backend Lab

![Compiler pipeline terminal preview](docs/inspect_compiler_pipeline_demo.svg)

`uTPU` is a focused ML systems project: a small PyTorch MLP subset lowers into a custom Graph IR, then into either a generated CUDA blocked-FC kernel or a custom uTPU ISA/RTL path. The point is not broad framework coverage. The point is compiler passes, backend lowering, measurement discipline, autotuning, and hardware-style verification on one narrow workload family.

## Scope

Supported:

- Batch-1 MLP-style `Linear -> ReLU -> Linear` flows.
- Pass-based Graph IR: shape inference, Linear+ReLU fusion, dead-code elimination, liveness memory planning, backend legality.
- INT4 blocked fully connected lowering for CUDA and uTPU program emission.
- CUDA runtime execution through generated NVRTC kernels.
- uTPU ISA generation plus Python ISA simulation and Verilog RTL simulation evidence.

Not claimed:

- General PyTorch compiler support.
- Transformer support.
- Production `torch.compile` backend.
- Physical board validation for Graph IR-generated programs.
- End-to-end speedup over PyTorch/cuBLAS on tiny benchmarks.

## Evidence-Backed Claims

- Compiler correctness: differential harness compares the NumPy Graph IR interpreter, CUDA compiled backend, uTPU quantized emulation, and a TorchInductor oracle when the local platform supports it. Evidence: [docs/EVIDENCE.md#differential-testing](docs/EVIDENCE.md#differential-testing), [firmware/host/differential_test_harness.py](firmware/host/differential_test_harness.py).

- CUDA performance engineering: calibrated CUDA-event data backs a schedule-aware analytical cost model and pruning policy for blocked-FC autotuning. The current measured-data replay profiles `4.92` of 16 schedules on average (`3.25x` search reduction), keeps every selected schedule within `1%` of exhaustive best, and lowers max replay regression from strict top-k's `5.90%` to `0.49%`. A small live CUDA smoke check also exercises exhaustive vs pruned tuning, but replay remains the primary quality claim. Evidence: [docs/EVIDENCE.md#cost-model-and-pruned-autotuner](docs/EVIDENCE.md#cost-model-and-pruned-autotuner), [firmware/host/cost_model.py](firmware/host/cost_model.py), [firmware/host/cuda_autotuner.py](firmware/host/cuda_autotuner.py).

- Hardware verification: Python ISA simulator and Verilog RTL produce bit-identical fetch bytes on two compiled fused MLP programs. Evidence: [docs/EVIDENCE.md#python-isa-simulator--rtl-bitmatch](docs/EVIDENCE.md#python-isa-simulator--rtl-bitmatch), [firmware/host/isa_simulator.py](firmware/host/isa_simulator.py), [rtl/tb/tb_fused_compressed_program.sv](rtl/tb/tb_fused_compressed_program.sv).

## Architecture

```text
PyTorch module
  -> torch.fx trace
  -> custom Graph IR
  -> pass pipeline
       shape inference
       Linear+ReLU fusion
       dead-code elimination
       liveness memory planning
       backend legality
  -> blocked-FC schedule/request layer
  -> backend lowering
       CUDA: NVRTC kernel + schedule autotuner + cost model
       uTPU: ISA program words + Python ISA sim + Verilog RTL sim
```

## Fast Demo

```bash
python examples/inspect_compiler_pipeline.py
```

This prints the FX graph, post-pass Graph IR, blocked-FC schedule, CUDA kernel metadata, uTPU instruction footprint, fallback lists, and writes `build/reports/compiler_introspection_tiny_mlp.json`.

Expected summary:

```text
FX graph: x -> fc1 -> relu -> fc2 -> output
Graph IR ops: linear_relu, linear
fallback_ops=[]
CUDA backend: blocked_fc_int4_kernel executable=True
uTPU ISA footprint: total_utpu_instruction_words=434
```

## Reproduce The Key Evidence

```bash
python -m pytest firmware/host/test_differential_harness.py -q
python -m pytest firmware/host/test_isa_simulator.py -q
python firmware/host/run_isa_rtl_bitmatch.py --output-json build/reports/isa_rtl_bitmatch_report.json --output-md build/reports/isa_rtl_bitmatch_report.md
python firmware/host/evaluate_pruned_autotuner.py --top-k 4 --output-json build/reports/pruned_autotuner_report.json
python examples/inspect_compiler_pipeline.py
```

For CUDA calibration and holdout validation, use the calibration scripts under `firmware/host/`; these generate local reports under `build/reports/`.

## Resume-Safe Wording

- Built a pass-based Graph IR compiler for a scoped MLP subset with shape inference, Linear+ReLU fusion, liveness memory planning, backend legality checks, and differential testing against reference/backend oracles.

- Built a cost-model-guided CUDA blocked-FC autotuner that prunes a 16-candidate schedule space to `4.92` measured candidates on calibrated replay (`3.25x` reduction), with all selected schedules within `1%` of exhaustive best and max replay regression under `0.5%`.

- Verified bit-accurate agreement between a Python uTPU ISA simulator and Verilog RTL on compiled fused MLP programs.

## Important Caveats

- The TorchInductor oracle is wired into the differential harness, but the current Windows artifact records a platform skip (`WinError 50`). Rerun on a supported Linux/WSL TorchInductor stack before claiming TorchInductor pass coverage.

- The uTPU differential backend in the compiler harness is software emulation. The separate ISA/RTL bitmatch claim is simulation-based, not board execution.
