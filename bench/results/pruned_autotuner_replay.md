# Pruned Autotuner Evaluation

- timestamp_utc: 2026-10-12T20:20:27.412311+00:00
- source_calibration_json: bench\fixtures\cost_model_calibration_seed.json
- calibrated_shapes: 24
- full_candidates_per_shape: 16
- profiled_candidates_per_shape: 5.125
- search_reduction_x: 3.12
- mean_quality_regression_pct: 0.04%
- p95_quality_regression_pct: 0.25%
- max_quality_regression_pct: 0.49%
- within_5pct_fraction: 1.00

| shape(in,out) | exhaustive_schedule | pruned_schedule | exhaustive_us | pruned_us | regression |
|---|---:|---:|---:|---:|---:|
| (128,16) | (256,4) | (32,1) | 7.476 | 7.513 | 0.49% |
| (32,16) | (256,8) | (32,4) | 6.741 | 6.757 | 0.25% |
| (16,512) | (256,4) | (256,1) | 6.685 | 6.695 | 0.15% |
| (16,16) | (32,2) | (32,2) | 6.839 | 6.839 | 0.00% |
| (16,64) | (256,8) | (256,8) | 6.759 | 6.759 | 0.00% |
| (16,256) | (32,1) | (32,1) | 6.783 | 6.783 | 0.00% |
| (32,64) | (32,2) | (32,2) | 6.767 | 6.767 | 0.00% |
| (32,256) | (32,1) | (32,1) | 6.846 | 6.846 | 0.00% |
| (32,512) | (32,1) | (32,1) | 6.823 | 6.823 | 0.00% |
| (64,16) | (32,8) | (32,8) | 6.650 | 6.650 | 0.00% |
