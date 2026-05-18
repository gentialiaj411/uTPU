# DEEP_CONTEXT
Optional deep reference for implementation details.

## Architecture Notes
- `graph_passes.py` now owns the explicit compiler-pass stage.
- `graph_reference_interpreter.py` executes Graph IR directly with NumPy as an independent oracle.
- `differential_test_harness.py` runs fixed-shape comparisons:
  - reference interpreter output
  - CUDA compiled output (or skip with reason)
  - uTPU quantized emulation output

## Differential Artifact Contract
- Output path: `build/reports/differential_test_report.json`
- Includes:
  - `tolerance`: `atol`, `rtol`
  - `environment`: python/numpy/torch versions and CUDA availability
  - `shapes`: entries for `(4,8,4)`, `(8,16,8)`, `(16,32,16)`
  - per-backend `status`, `max_abs_error`, `max_rel_error`, and execution mode

## Test Meaning Map
- `test_differential_harness.py`:
  - writes report artifact
  - asserts backend entries and tolerance checks
  - permits CUDA skip with explicit CUDA-related reason
  - requires uTPU emulation comparison pass

## Known Limits
- uTPU differential comparison is software emulation, not board execution.
- Subset remains blocked-FC MLP only.
