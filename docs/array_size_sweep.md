# Array Size Sweep (8x8 vs 16x16)

This flow builds two Vivado projects with different top-level `ARRAY_SIZE` generics and writes comparable reports.

## Run

From repo root:

```powershell
cd C:\Users\bhask\Documents\PROJECTS\uTPU
vivado -mode batch -source scripts/array_size_sweep.tcl
```

## Output

Reports are generated here:

- `build/reports/array_8/`
- `build/reports/array_16/`

Each folder contains:

- `utilization_synth.rpt`
- `utilization_synth_hier.rpt`
- `timing_synth.rpt`
- `utilization_impl.rpt`
- `utilization_impl_hier.rpt`
- `timing_impl.rpt`
- `drc_impl.rpt`

## What To Check

1. `utilization_impl_hier.rpt`
   - `u_unified_buffer` should show BRAM usage (`RAMB18`/`RAMB36`) and much lower LUTs than pre-fix builds.
2. `utilization_impl.rpt`
   - Compare total LUT/FF/DSP/BRAM between 8 and 16.
3. `timing_impl.rpt`
   - Confirm WNS/TNS timing still closes for 16.
4. `drc_impl.rpt`
   - Confirm no new critical implementation issues.

## Quick Compare Commands

```powershell
Select-String -Path build\reports\array_8\utilization_impl_hier.rpt -Pattern "u_unified_buffer"
Select-String -Path build\reports\array_16\utilization_impl_hier.rpt -Pattern "u_unified_buffer"
Select-String -Path build\reports\array_8\timing_impl.rpt -Pattern "WNS|TNS"
Select-String -Path build\reports\array_16\timing_impl.rpt -Pattern "WNS|TNS"
```
