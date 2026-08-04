# uTPU writeup — causal chain (measured numbers only)

Every figure below traces to a committed artifact. Workload and harness are
named per standing rule 5 (never report a mode without labeling it). Roofline /
design-space completion is **stubbed** pending Track 4 grid finish.

## 1. The problem the measurements forced

Early systolic characterization showed single-digit array duty on realistic
batch sizes. That is a symptom, not a diagnosis.

**Harness / workload.** Isolated blocked-FC GEMM, iverilog RTL,
`firmware/host/run_cycle_attribution.py`, artifact
[`bench/results/cycle_attribution.json`](../bench/results/cycle_attribution.json).

On **64×64 B=48 N=16** (fast-UART TB, on-chip core cycles — **not** board
wall-clock at 115200 baud):

| Group | Share of START→HALT |
|---|---:|
| Instruction stream (fetch/decode + STORE/BSTORE framing) | **~66%** |
| `result_fetch` | ~27% |
| `compute` | ~5.7% |
| buffer→PE `load` | **~1.1%** |

Duty-cycle talk that starts from “make the PE busier” without this split is
aiming at the wrong 1%.

## 2. Why capacity, not PE overlap, is the real hinge

**Harness / workload.** Host program-word composition over fused MNIST and
board-fit shapes — [`bench/results/program_word_composition.json`](../bench/results/program_word_composition.json).

Fused MNIST is **~87.2%** embedded BSTORE payload (1296 / 1486 words). Capstone
FC1 payload alone is **14144 words**, which already exceeds the historical
`PROG_DEPTH=8192` instruction BRAM. Part A’s control compression therefore
cannot unlock real-model fit while weights live in the program image: the
program is **payload-bound**, not control-bound.

## 3. Three levers, sized against multi-layer inference

Isolated-GEMM Amdahl (≤~3× if the instruction stream vanished) is a harness
ceiling. Multi-layer fused MNIST is the grading workload.

| Lever | What it does | Multi-layer evidence |
|---|---|---|
| Requant / finalize rightsizing | Cut DSP so the part closes | **192 → 72 DSP** on Artix-7 A7-100T (`requant_rightsizing_synth.json`) |
| Fmax / pipe depth | Raise clock under positive WNS | Shipping **12 ns / ~83 MHz** (WNS=+0.271, thin); demonstrated ceiling **100 MHz** (WNS=+**0.012**, marginal) — `design_space_sweep.json` / `timing_closure_sweep.json` |
| Instruction + buffer capacity | Fit real programs | Board-fit **4/14 → 10/14** shapes at `PROG_DEPTH=65536` (`board_fit_audit.json`) |
| BSTORE write-arm widen | Cut payload upload cycles | Fused MNIST e2e **6523 → 3445** cycles ≈ **1.89×** (`cycle_attribution_mnist.json`, `bstore_path_measure.json`) |
| Buffer-resident weights (A5) | Amortize BSTORE out of steady state | Cold **3445** → steady-state **1261** cycles; compute share **~20.6% → ~56.2%**, **bit-exact** (`steady_state_attribution.json`) |

Occupancy / compute-share numbers are labeled **cold** vs **steady-state** and
must stay that way.

## 4. Two hypotheses left visible because they failed honestly

### 4a. PE weight double-buffering / concurrent LOAD+COMPUTE

**Disproven for this RTL.** Attribution puts buffer→PE `load` at **~1.1%** of
on-chip cycles on the flagship isolated shape. Overlapping that arm cannot move
duty cycle. Approach A weight-overlap remained cycle-neutral in measurement.
Redirect: cut the instruction/payload stream, not the load path.
Evidence: `cycle_attribution.json` verdict
`confirmed_fetch_stream_dominates_load_negligible`.

### 4b. Part A descriptor ISA as a capacity unlock *by itself*

**Disproven as a capacity story while weights stay in-image.** Composition shows
payload dominates; FC1’s 14144-word payload alone exceeds `PROG_DEPTH=8192`.
Part A’s control compression is still useful for cycle Amdahl **after**
buffer-resident weights remove payload from the program image — that is why
Part A stays **parked** until steady-state is real (it now is, in sim).
Evidence: `program_word_composition.json`, `docs/descriptor_isa_part_a_design.md`.

## 5. Bugs that are verification evidence

### 5a. Upload-length width bug (`PROG_DEPTH[15:0]` truncate)

At `PROG_DEPTH=65536` (`0x10000`), comparing the UART length field to
`PROG_DEPTH[15:0]` truncates the limit to **0**. Every nonzero upload was
rejected; Vivado then constant-folded `prog_len` / `upload_count` / the FSM
(first “close” was invalid: RAMB36≈2, LUT~11k). Fix:
`UPLOAD_LEN_MAX = min(PROG_DEPTH, 65535)` plus width-safe addressing.
Caught because synthesis collapsed the design — not because a functional depth
test existed first. Class closed by parameterized iverilog smoke
(`prog_depth_smoke.json`, `docs/descriptor_isa_part_a_design.md` §0a).

### 5b. BSTORE weight-region aliasing (buffer-resident weights)

Making weights persist across inferences exposed a correctness defect that cold
runs hide: fused-MNIST BSTORE tiles **reuse** the same buffer addresses
(e.g. repeated stores to 128). After one cold inference the buffer holds only
the last tile; a control-only replay then LOADs the wrong weights —
cycle count looked amortized (~1261) but logits mismatched.

Fix: remap each BSTORE payload into a unique durable span in
`[0, 14144)` and relocate RUN destinations into `[14144, 16384)` before A5
fill. After remap, A5-once + control-only is **bit-exact** for N=3
inferences (`steady_state_attribution.json`). This is a real
verification finding, not a footnote — see CLAIMS_MATRIX row.

## 6. Predictability: CUDA vs uTPU cost models

Identical held-out methodology (80/20 by shape, refit TRAIN only). **Schedule
menus differ:**

| Backend | Candidates / shape | Held-out log_R² | MAPE | Selection regret mean / max |
|---|---:|---:|---:|---:|
| CUDA blocked-FC | **16** | 0.926 | 14.32% | **5.21% / 11.90%** |
| uTPU RTL cycles | **5** | 0.924 | 10.04% | **0.00% / 0.00%** |

Zero uTPU regret is scoped to the **5-candidate** deterministic menu
`(batch, hoist)`, not CUDA’s 16-way autotuner space. Details:
[`docs/COSTMODEL_COMPARISON.md`](COSTMODEL_COMPARISON.md),
[`bench/results/utpu_cycle_model_heldout.json`](../bench/results/utpu_cycle_model_heldout.json),
[`bench/results/cost_model_heldout.json`](../bench/results/cost_model_heldout.json).

## 7. Determinism vs GPU tail latency

**Harness.** FPGA: iverilog blocked-FC `(M=32,K=32)`, 37 inputs, cycle variance
**0** (2784 cycles every trial) —
[`latency_determinism.json`](../bench/results/latency_determinism.json).
GPU: same logical GEMV, N=10000, CUDA events —
[`latency_determinism_vs_gpu.json`](../bench/results/latency_determinism_vs_gpu.json).
Plot: [`docs/latency_determinism_vs_gpu_logx.png`](latency_determinism_vs_gpu_logx.png).

**Clock basis (reconciled with the sweep).**

| Basis | Fmax | WNS | Role | FPGA p50 wall | Median-latency loss vs GPU p50 |
|---|---:|---:|---|---:|---:|
| **Shipping** | **~83.3 MHz** (12 ns) | **+0.271** (thin) | **Headline** | 33408 ns | **~1.89×** |
| Demonstrated ceiling | 100 MHz (10 ns) | **+0.012** (marginal) | Ceiling only — quote WNS | 27840 ns | ~1.57× |

Claim = **bounded jitter** + median-latency loss. **Not** “FPGA is faster.”
Scopes differ (RTL-sim cycles @ stated clock vs live GPU kernel events).

## 8. Roofline / design-space (stub — Track 4 in flight)

Serial Vivado sweep at `PROG_DEPTH=65536`, `QUANTIZER_PIPE_DEPTH=3` is running
one point at a time. Artifact schema v2 already separates:

- per-period **timing evidence** (period, WNS, util, `margin_class`)
- **throughput_by_config** — one GOP/s point per `(N, CDW, MB)` from the
  tightest closing period

Peak ≠ achieved (achieved = peak × occupancy; occupancy from bit-exact
steady-state compute share when available). Shipping rationale and binding
resources land in [`docs/HARDWARE_DESIGN_SPACE.md`](HARDWARE_DESIGN_SPACE.md)
and [`bench/results/design_space_sweep.json`](../bench/results/design_space_sweep.json)
when `status` flips from `partial` to `complete`. Until then, cite only closed
points and the explicit `not_yet_attempted` list.

## 9. What is still open

- On-board FPGA execution (P0) — simulation ≠ silicon.
- Track 4 full grid + optional 10 ns MB=48 seed reproducibility.
- Track 1 final `BUFFER_SIZE=16384` re-synth (queued behind the grid).
- Part A descriptor ISA — parked; post–steady-state Amdahl ≈1.22× on cold
  multi-layer until control compression lands.

## 10. Reproduction map

See [`docs/REPRO.md`](REPRO.md). Narrow gates used above:

```bash
python firmware/host/run_cycle_attribution.py
python firmware/host/run_program_word_composition.py
python firmware/host/run_steady_state_attribution.py
python -m pytest firmware/host/test_utpu_cycle_model_heldout.py -q
python -m pytest firmware/host/test_latency_analysis.py -q
```
