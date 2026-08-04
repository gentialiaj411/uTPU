# Cost-model comparison: CUDA vs uTPU

Held-out predictability on unseen layer shapes. Same split discipline
(deterministic 80/20 by shape, refit on TRAIN only). CUDA artifact:
`bench/results/cost_model_heldout.json`. uTPU artifact:
`bench/results/utpu_cycle_model_heldout.json` (predicts RTL
`total_program_cycles` from iverilog / cached sims).

| Metric (held-out TEST) | CUDA (µs latency) | uTPU (RTL cycles) |
|---|---:|---:|
| log_R² | 0.926 | 0.498 |
| MAPE | 14.32% | 36.18% |
| Selection regret mean | 5.21% | 0.0% |
| Selection regret max | 11.90% | 0.0% |
| Top-1 schedule accuracy | 0.0 | 1.0 |

**Finding:** uTPU hardware is cycle-deterministic, but the *analytical*
cycle model is **not** trivially accurate on held-out shapes
(`trivially_accurate_due_to_determinism=false` in the artifact) — MAPE
36% on TEST vs 6% on TRAIN shows shape generalization is the hard part.
Selection regret is 0% on the small multi-schedule held-out set (top-1
always hit), which is a different question from latency MAPE.

CUDA numbers are wall-clock on WSL2 + RTX 5070 calibration JSON; uTPU
numbers are on-chip RTL program cycles (fast-UART TB), not board
wall-clock. Do not cross-compare absolute units — only predictability
structure.
