# Current-RTL 8x8 INT8 timing attempt log

Git SHA: `48fe65a2a98775c33d3e84e268fad06502e6efbc`
Vivado: `2025.2`
Part: `xc7a100tcsg324-1`
Top: `top`
Fixed generics: `PROG_DEPTH=8192 COMPUTE_DATA_WIDTH=8 ACCUMULATOR_DATA_WIDTH=32 ARRAY_SIZE=8 BUFFER_SIZE=4096 EXT_ADDR_EN=1`

## Baseline 100 MHz sweep

1. `MAX_BATCH_COUNT=16`, `clock=10.0 ns`
   - Change: none, current RTL
   - Result: route clean, `.bit` generated
   - WNS: `-6.379 ns`
   - Kept/reverted: kept as measured artifact

2. `MAX_BATCH_COUNT=8`, `clock=10.0 ns`
   - Change: none, current RTL
   - Result: route clean, `.bit` generated
   - WNS: `-6.292 ns`
   - Kept/reverted: kept as measured artifact

3. `MAX_BATCH_COUNT=4`, `clock=10.0 ns`
   - Change: none, current RTL
   - Result: route clean, `.bit` generated
   - WNS: `-5.714 ns`
   - Kept/reverted: kept as measured artifact

## Critical-path diagnosis

4. `MAX_BATCH_COUNT=4`, post-route timing report
   - Change: none
   - Result: top 3 worst paths all pass through `u_quantizer_array`
   - Worst path summary: `requant_multiplier_latched_reg[*]` to `compute_to_buffer_reg[*]`, logic `8.449 ns`, route `7.146 ns`, total `15.595 ns`
   - Kept/reverted: kept as diagnosis artifact

## Relaxed-clock closure sweep

5. `MAX_BATCH_COUNT=4`, `clock=16.0 ns`
   - Change: none
   - Result: route clean, `.bit` generated
   - WNS: `-0.704 ns`
   - Kept/reverted: kept as measured artifact

6. `MAX_BATCH_COUNT=4`, `clock=18.0 ns`
   - Change: none
   - Result: route clean, `.bit` generated
   - WNS: `-0.397 ns`
   - Kept/reverted: kept as measured artifact

7. `MAX_BATCH_COUNT=4`, `clock=20.0 ns`
   - Change: none
   - Result: route clean, `.bit` generated, timing met
   - WNS: `+0.416 ns`
   - Kept/reverted: kept as restored current-RTL closed configuration

## 10 ns flow-opt recovery

8. `MAX_BATCH_COUNT=4`, `clock=10.0 ns`, `retiming=1`, `post_place_phys_opt=1`, `post_route_phys_opt=1`, `synth_strategy=Flow_PerfOptimized_high`, `impl_strategy=Performance_Explore`
   - Change: Tcl-only flow knobs, no RTL change
   - Result: blocked before synthesis
   - Error: `Failed to install all user apps` / `load_features`
   - Kept/reverted: no design result produced

## Verification status

- No RTL changes were made during this timing-restoration pass.
- RTL verification was rerun after the artifact update in this pass:
  - `run_isa_rtl_bitmatch.py`: pass, `all_isa_rtl_bitmatch=true`
  - `run_isa_rtl_residual_bitmatch.py`: pass, `all_isa_rtl_bitmatch=true`
  - `run_pe_packed_pair_sim.py`: pass
  - `run_pe_array_packed_sim.py`: pass artifact, odd-size guard compile intentionally trips in the secondary negative test
  - `run_pe_array_packed_hardened_sim.py`: pass
  - `run_packed_array_cycle_compare.py`: pass, cycle counts unchanged (`8x8 9/23`, `16x16 17/47`)
  - `run_top_packed_smoke.py`: pass
  - `pytest firmware/host/test_batched_gemm.py firmware/host/test_batched_gemm_rtl_artifact.py firmware/host/test_batched_gemm_rtl_sweep.py firmware/host/test_packed_dsp_sims.py -q`: `13 passed`
