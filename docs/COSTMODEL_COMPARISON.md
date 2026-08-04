# CUDA vs uTPU cost-model predictability

Side-by-side held-out generalization for the CUDA blocked-FC latency cost
model and the uTPU RTL cycle cost model. Both use the same methodology:
deterministic 80/20 split by `(in_features, out_features)`, refit on TRAIN
only, evaluate log-R² / MAPE / selection regret on unseen shapes.

## Candidates per shape (schedule space)

| Backend | Candidates per shape | What a "schedule" is |
|---|---:|---|
| **CUDA** blocked-FC | **16** | Autotuner menu (threads/block × unroll × related CTA knobs) on the calibrated grid |
| **uTPU** batched GEMM RTL | **5** | `(batch_size ∈ {1,4,16}, hoist_tile_payloads ∈ {0,1})` with B=1 hoist omitted → 5 legal cells |

uTPU's held-out selection regret of **0.00% / 0.00%** is therefore scoped to a
**5-candidate** space, not CUDA's **16-candidate** space. Do not read the two
regret numbers as an apples-to-apples "same menu, different backend" score.
They share methodology (refit TRAIN-only; argmin predicted; regret vs measured
oracle) but the menus differ by ~3×.

## Held-out metrics

| Backend | Target quantity | Cands / shape | Held-out log_R² | Held-out MAPE | Selection regret mean / max | Top-1 | Finding |
|---|---|---:|---:|---:|---:|---:|---|
| **CUDA** blocked-FC | measured kernel µs (noisy GPU) | 16 | 0.926 | 14.32% | **5.21% / 11.90%** | 0.00 | Bounded regret under GPU noise; top-1 not claimed. |
| **uTPU** batched GEMM RTL | `TOTAL_PROGRAM_CYCLES` (iverilog) | 5 | 0.924 | 10.04% | **0.00% / 0.00%** | 1.00 | Zero regret on the **5-candidate deterministic** menu. Absolute MAPE is residual analytical-model error, not timing jitter. |

## Interpretation

If the schedule spaces were comparable, the honest contrast would be:

> Identical held-out methodology, two backends: **zero** selection regret on the
> deterministic uTPU target vs **5.21% mean / 11.90% max** on the GPU.

Because uTPU evaluates only 5 candidates per shape (vs CUDA's 16), that sentence
is scoped: zero regret is real evidence of determinism on the measured menu, not
proof the same fitter would stay at zero regret on a 16-way uTPU autotuner menu
that does not yet exist.

Absolute cycle MAPE on uTPU (~10%) remains non-zero because the fitted feature
model approximates the RTL control/compute mix; closing it further is a
structural-model task, not a measurement-noise task.

## Sources

- CUDA artifact: [`bench/results/cost_model_heldout.json`](../bench/results/cost_model_heldout.json)
  — harness [`firmware/host/run_cost_model_heldout.py`](../firmware/host/run_cost_model_heldout.py)
  — per held-out shape `n_schedules=16`
- uTPU artifact: [`bench/results/utpu_cycle_model_heldout.json`](../bench/results/utpu_cycle_model_heldout.json)
  — harness [`firmware/host/run_utpu_cycle_model_heldout.py`](../firmware/host/run_utpu_cycle_model_heldout.py)
  — `grid.schedules` length 5; per held-out shape `n_schedules=5`
- uTPU ground truth: iverilog batched-GEMM sims (`TOTAL_PROGRAM_CYCLES`), cached under
  `build/utpu_cycle_cache/`; seeded from
  [`bench/results/systolic_characterization.json`](../bench/results/systolic_characterization.json)
  where keys match.

## Scope / limitations

- Units differ: CUDA reports microseconds; uTPU reports RTL program cycles.
  Field names in the uTPU artifact mirror the CUDA schema for tooling parity.
- No claim that uTPU wall-clock beats CUDA; this comparison is **predictability
  under held-out shapes** only.
- Expanding the uTPU schedule menu (more batch/hoist/tiling knobs) would require
  a fresh held-out regen before claiming CUDA-comparable regret.
