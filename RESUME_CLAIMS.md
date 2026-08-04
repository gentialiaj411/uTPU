# RESUME_CLAIMS.md

Every bullet names the **artifact** and the **workload/harness** it was measured on.
Dropped: claims whose artifacts are absent (`fuzzer_report.json`) or superseded
(pre-rightsizing **50 MHz / DSP=192** as the *shipping* close; use **~83 MHz / DSP=72**).

## Flagship bullets (submittable)

- Built a PyTorch-to-accelerator compiler that lowers FX graphs into a pass-based Graph IR with NVRTC CUDA and a custom 16-bit SystemVerilog ISA; INT8/INT4 MLPs are bit-identical to a NumPy oracle across tiling shapes, and the Python ISA simulator is byte-exact with iverilog RTL on the bitmatch corpus.
  - Artifacts: `bench/results/tiling_correctness.json` (blocked-FC INT4 tiling sweep), `build/reports/isa_rtl_bitmatch_report.json` (ISA↔RTL), `bench/results/real_model_end_to_end.json` (ResNet-18 CUDA graph path).
  - Workloads: blocked-FC tiling shapes; fused MLP bitmatch programs; ResNet-18 on CUDA graph executor (not uTPU full-model).

- Deployed INT8 accuracy **97.33%** (float **97.30%**) on a 14×14 MNIST `196×256×10` MLP under the hardware-actual integer contract (int matmul → per-layer fixed-point requant); INT4 **58.32%**. Bit-exact host integer reference ↔ ISA ↔ RTL on representative shapes / FC2; FC1 ISA-only.
  - Artifact: `bench/results/real_model_accelerator.json`.
  - Workload: 14×14 MNIST PTQ capstone; simulation + host-reference (not on-silicon).

- Closed Artix-7 A7-100T (`xc7a100tcsg324-1`) post-route at **~83.3 MHz** (`12 ns`, **WNS=+0.271 ns**) after requant rightsizing cut DSP **192→72**; demonstrated **100 MHz** ceiling only (**WNS=+0.012 ns**, marginal).
  - Artifacts: `bench/results/requant_rightsizing_synth.json`, `bench/results/design_space_sweep.json` (`shipping_point` / `demonstrated_fmax_ceiling`).
  - Workload/harness: N=8 INT8, `MAX_BATCH_COUNT=48`, Vivado P&R (not board execution).

- Cut launch-bound CUDA dispatch by collapsing multi-op chains into one NVRTC region-kernel launch — **62.5% pooled launch reduction** (6 fused vs 16 op-by-op) with bit-exact oracle parity; latency-% remains `[needs-locked-clock-artifact]` on unlocked WSL2.
  - Artifact: `bench/results/megakernel_payoff.json`.
  - Workload: six measured micro-workloads on WSL2 + RTX 5070 (count-based claim only).

## Hardware arc (simulation + synth)

- Instruction-stream dominates on-chip cycles: **66%** stream / **1.1%** LOAD on isolated GEMM **64×64 B=48 N=16** (fast-UART TB) — parks concurrent LOAD+COMPUTE.
  - Artifact: `bench/results/cycle_attribution.json`.

- `BSTORE_WIDTH=8` delivers **~1.89×** end-to-end cycle reduction on multi-layer fused MNIST (6523→3445).
  - Artifacts: `bench/results/bstore_path_measure.json`, `bench/results/cycle_attribution_mnist.json`.
  - Workload: fused MNIST case1, N=16 fast-UART TB.

- Buffer-resident weights (A5): steady-state compute share **~20.6%→56.2%**, mean **1261** cycles, bit-exact vs cold.
  - Artifact: `bench/results/steady_state_attribution.json`.
  - Workload: remapped fused MNIST, iverilog.

- Instruction BRAM capacity: **4/14** shapes at `PROG_DEPTH=1024` → **10/14** at shipping **`PROG_DEPTH=65536`**.
  - Artifacts: `bench/results/board_fit_audit.json`, `bench/results/prog_depth_sweep.json`.

- uTPU cycle cost model held-out: **log R² 0.924**, **MAPE 10.04%**, **0% selection regret** on a **5-candidate** menu (CUDA comparison is 16-candidate, mean regret 5.21%).
  - Artifacts: `bench/results/utpu_cycle_model_heldout.json`, `bench/results/cost_model_heldout.json`, `docs/COSTMODEL_COMPARISON.md`.

- Determinism: **zero** FPGA/RTL cycle variance; median wall-clock **~1.89×** GPU p50 at shipping ~83 MHz (loss, not speedup); FPGA p99=p50 flat vs GPU p99/p99.9 tails in artifact.
  - Artifacts: `bench/results/latency_determinism_vs_gpu.json`, `bench/results/latency_determinism.json`.
  - Workload: RTL sim cycles@shipping clock vs GPU NVRTC blocked-FC INT4 events.

- UART host→device→host path validated in iverilog (byte-exact to ISA sim); `on_silicon.status=simulation`.
  - Artifact: `bench/results/uart_preboard_roundtrip.json`.
  - Workload: lowered 8×8 INT8 program via `tb_uart_replay.sv`.

## Compiler / CUDA (still current)

- Pass-based Graph IR: shape inference, Linear+ReLU fusion, DCE, liveness memory planning, backend legality.
  - Evidence: `firmware/host/test_graph_passes.py` (blocked-FC MLP).

- NumPy Graph IR interpreter as independent oracle.
  - Evidence: `firmware/host/test_reference_interpreter.py`.

- Differential harness for supported ops with explicit host fallback.
  - Evidence: `build/reports/differential_test_report.json`, `firmware/host/test_differential_harness.py`.

- `conv2d` via im2col to GEMM datapath; CIFAR-10 all-conv CNN INT8 **69.11%** vs float **70.45%** on deployed integer reference (128-image backend parity subset).
  - Artifacts: `bench/results/utpu_small_cnn_validation.json`, `bench/results/utpu_conv2d_validation.json`.

- Transformer attention GEMM hybrid: Q/K/V, QKᵀ, AV, out-proj bit-exact; softmax/LN/residual on host.
  - Artifact: `bench/results/utpu_attention_hybrid.json`.

- Per-output-channel requant co-design (capability, not accuracy win — INT8 flat).
  - Artifact: `bench/results/real_model_accelerator.json`; tests: `firmware/host/test_blocked_fc_requant.py`.

- Scheduler + allocator: sim-cycle reduction **4.67%**; RTL cross-check within ±2.0% permille on `(M=32,K=32)` with naive===scheduled fetch bytes.
  - Artifacts: `bench/results/scheduler_cycles.json`, `bench/results/scheduler_rtl_crosscheck.json`.

- CUDA cost-model consumed at runtime (`schedule_source="cost_model"`); held-out CUDA log_R² **0.926**, MAPE **14.32%**, mean regret **5.21%**.
  - Artifacts: `bench/results/selection_ab.json`, `bench/results/cost_model_heldout.json`.

- CUDA cost-model regression gate: median abs-% error **8.98%**, p95 **15.71%** over 24 shapes (measured-data replay).
  - Artifact: `bench/results/cost_model_regression.json`.

- 2-PE K-split FC1 path (simulator only): **24.1%** parallel cycle reduction on `case2_multi_k`.
  - Artifact: `bench/results/multi_pe_sim.json`.

- Superoptimizer on 8 matmul-chain graphs: **100%** win rate, **87.5%** median modeled FLOP reduction on wins (sim-only).
  - Artifact: `bench/results/superopt_payoff.json`.

- Transformer-block host validation + INT4 weight-memory reduction **71.875%**.
  - Evidence: `firmware/host/test_transformer_integration.py`, `build/reports/transformer_parity_report.json`, `build/reports/int4_quantization_report.json`.

- Calibrated CUDA cost model + pruned autotuner (replay): log_R² ~0.936, profiles ~4.9/16 schedules (~3.3× search cut).
  - Evidence: `build/reports/cost_model_calibration.json`, `build/reports/pruned_autotuner_report.json`.

## Explicitly not claimed / dropped

- On-board FPGA execution (P0 open).
- FPGA faster than GPU.
- Shipping at 100 MHz without quoting marginal WNS.
- uTPU 0% regret as CUDA-menu-comparable (5 vs 16 candidates).
- Metamorphic fuzzer resume bullet — `bench/results/fuzzer_report.json` absent on this host.
- Pre-rightsizing **50 MHz / DSP=192** as current shipping close (kept only as provenance in `baseline_8x8_current_rtl_synth.json` / rightsizing table).

## Caveats

- Simulation ≠ silicon. UART, cycle attribution, and latency-vs-GPU FPGA arm are iverilog / clock-converted.
- Do not cite 8×8 MNIST `proxy_quant_acc≈0.92` as accelerator accuracy; use 14×14 PTQ **97.33%**.
- Attention hybrid is host/accelerator split — not “attention on the accelerator.”
- GPU latency percentages on unlocked WSL2: `[needs-locked-clock-artifact]`.
