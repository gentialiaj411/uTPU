## Packed-DSP Vivado synth run-note (user-executed; agent does not run Vivado)

**Status:** scripts + JSON writer prepared; all DSP/timing numbers below are **UNCONFIRMED** until you run Vivado on a host with a license.

**Part:** `xc7a100tcsg324-1` (Arty A7-100T Rev E)  
**Flow:** `scripts/synth_packed_dsp.tcl` (mirrors `program_arty_a7_revE.tcl` report prefix pattern)  
**Artifact after parse:** `bench/results/packed_dsp_synth.json` (schema matches `p4_2_vivado_reports.json`)

### Host commands (four configs)

Run from repo root. Each invocation synth+impl+writes reports under `build/reports/` with the given prefix. **Do not program the board** (`do_program=0`).

```bash
# 1) Baseline pe_array via shipping top — 8×8 INT8
vivado -mode batch -source scripts/synth_packed_dsp.tcl \
  -tclargs do_program 0 top_name top report_prefix packed_baseline_8x8_int8 \
  ARRAY_SIZE 8 COMPUTE_DATA_WIDTH 8 ACCUMULATOR_DATA_WIDTH 32 BUFFER_SIZE 4096 EXT_ADDR_EN 1

# 2) Baseline pe_array via shipping top — 16×16 INT8 (may hit DSP ceiling; UNCONFIRMED)
vivado -mode batch -source scripts/synth_packed_dsp.tcl \
  -tclargs do_program 0 top_name top report_prefix packed_baseline_16x16_int8 \
  ARRAY_SIZE 16 COMPUTE_DATA_WIDTH 8 ACCUMULATOR_DATA_WIDTH 32 BUFFER_SIZE 4096 EXT_ADDR_EN 1

# 3) Packed array synth top — 8×8 INT8
vivado -mode batch -source scripts/synth_packed_dsp.tcl \
  -tclargs do_program 0 top_name top_packed report_prefix packed_array_8x8_int8 \
  ARRAY_SIZE 8 COMPUTE_DATA_WIDTH 8 ACCUMULATOR_DATA_WIDTH 32

# 4) Packed array synth top — 16×16 INT8
vivado -mode batch -source scripts/synth_packed_dsp.tcl \
  -tclargs do_program 0 top_name top_packed report_prefix packed_array_16x16_int8 \
  ARRAY_SIZE 16 COMPUTE_DATA_WIDTH 8 ACCUMULATOR_DATA_WIDTH 32
```

After all four runs complete, parse reports into JSON:

```bash
python firmware/host/write_packed_dsp_synth_json.py
```

### Hypothesized DSP counts (UNCONFIRMED)

| Config | Hypothesis | Rationale |
|--------|------------|-----------|
| `packed_baseline_8x8_int8` | **~64 DSP** | One DSP48 per MAC; matches existing `widened_int8` run (64/240) |
| `packed_baseline_16x16_int8` | **~256 DSP** | Exceeds Arty 240-DSP budget — expect synth/route failure or over-util |
| `packed_array_8x8_int8` | **~32 DSP** | Two INT8 MACs per DSP48 via operand packing |
| `packed_array_16x16_int8` | **~128 DSP** | Half of 256; target fits in 240 with margin |

These are engineering estimates only. The authoritative numbers are `dsp_used` in the generated JSON after Vivado.

### What WNS ≥ 0 proves

- **WNS ≥ 0** with `all_paths_met: true` means the placed-and-routed netlist meets the XDC clock constraint at the reported corner — it is a **synthesis/timing-closure** claim for that config on that part, not board execution or functional correctness.
- Comparing `dsp_used` baseline vs packed on the same part proves whether the packed topology actually saves DSPs in implementation (not in iverilog sim).
- A negative WNS on the 16×16 baseline config would confirm the known DSP/timing ceiling rather than a packed-array bug.

### Report files expected per run

- `build/reports/<report_prefix>_timing_summary.rpt`
- `build/reports/<report_prefix>_utilization.rpt`
- `build/reports/<report_prefix>_route_status.rpt`
- `build/reports/<report_prefix>.bit` (optional for evidence)

### Scope fence

This run-note does **not** claim FPGA demo throughput, power, or on-board execution. iverilog sim bit-exact results are separate artifacts (`pe_array_packed_sim.json`, `pe_array_packed_hardened.json`, etc.).
