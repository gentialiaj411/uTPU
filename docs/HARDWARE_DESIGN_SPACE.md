# Hardware design space (Artix A7-100T)

_Generated 2026-08-04T14:56:59.162408+00:00 · git `47865fa`_

## Scope

Full-top Vivado synth/impl (route, no bitstream) on `xc7a100tcsg324-1` at `PROG_DEPTH=65536`, `QUANTIZER_PIPE_DEPTH=3`, `BUFFER_SIZE=4096`.

Grid: `ARRAY_SIZE ∈ {4,8}` × `COMPUTE_DATA_WIDTH ∈ {4,8}` × `MAX_BATCH_COUNT ∈ {4,16,48}` × `period ∈ {20,15,12,10}` ns, plus a `16×16 INT4` Pareto attempt.

## Occupancy (peak ≠ achieved)

- **occupancy** = `0.205806`
- **occupancy_source** = cycle_attribution_mnist.json post-widen placeholder (compute 709/3445); steady_state_attribution.json absent
- Placeholder cold-path compute share until buffer-resident / steady-state attribution lands. NOT peak; applied as achieved = peak * occupancy.
- **peak_gops** = `ARRAY_SIZE² × 2 × Fmax_GHz`
- **achieved_gops** = `peak_gops × occupancy` (never report peak as achieved)

## Shipping point rationale

Chosen shipping point: **N=8 INT8 MAX_BATCH_COUNT=48 @ 20.0 ns** (prefix `prog_depth_pd65536_buf4096_mb48_clk20_pd3`).

Why:
1. **Accuracy** — INT8 per-layer accuracy 97.33% vs INT4 58.32% (`real_model_accelerator.json`).
2. **Board fit** — closes on xc7a100t with PROG_DEPTH=65536 instruction BRAM and remaining LUT/DSP/BRAM headroom for the MNIST/FC class.
3. **Batch ceiling** — MAX_BATCH_COUNT=48 is the largest closing batch from the timing-closure / LUT bisect path (mb64 LUT-oversubscribes).
4. **Clock** — 20 ns (50 MHz) is the shipping constraint with positive WNS; tighter periods are exploration-only.

Evidence: WNS=2.789 ns, Fmax≈58.10237638719424 MHz, LUT=41631/63400, DSP=72/240, BRAM=49/135, peak=7.437104177560863 GOP/s, achieved=1.530597057152584 GOP/s.
Highest util resource at shipping close: **lut** (not oversubscribed).

## Accuracy–throughput Pareto

- Point A (INT8): accuracy=97.33%, achieved_gops=1.530597057152584
- Point B (16×16 INT4): **did not close** or not yet attempted — see binding table below.

## Binding resource by corner

| N | CDW | MB | period_ns | status | WNS | Fmax_MHz | LUT | DSP | BRAM | peak_GOP/s | achieved_GOP/s | binder |
|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---|
| 8 | 8 | 16 | 12.0 | missing_reports | None | None | None/None | None/None | None/None | None | None | missing_reports |
| 8 | 8 | 16 | 10.0 | missing_reports | None | None | None/None | None/None | None/None | None | None | missing_reports |
| 8 | 8 | 48 | 20.0 | closed | 2.789 | 58.10237638719424 | 41631/63400 | 72/240 | 49/135 | 7.437104177560863 | 1.530597057152584 | lut |
| 8 | 8 | 48 | 15.0 | missing_reports | None | None | None/None | None/None | None/None | None | None | missing_reports |
| 8 | 8 | 48 | 12.0 | missing_reports | None | None | None/None | None/None | None/None | None | None | missing_reports |
| 8 | 8 | 48 | 10.0 | missing_reports | None | None | None/None | None/None | None/None | None | None | missing_reports |

## Summary counts

- closed: **1**
- failed: **0**
- missing/skipped: **5**

## Artifacts

- JSON: `bench/results/design_space_sweep.json`
- Plot: `docs/design_space_roofline.png`
- Runner: `firmware/host/run_design_space_sweep.py`
- TCL: `scripts/synth_design_space_point.tcl`

