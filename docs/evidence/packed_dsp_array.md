# Packed-DSP array evidence (staging writeup — not a resume claim)

**Evidence level:** simulated / CI-validated only. No Vivado execution in agent environment; no on-board execution.

## Sharing scheme

`pe_array_packed` pairs adjacent weight columns into one DSP48 MAC per even mesh column:

- Shared streamed activation at even columns; `pe_act_delay` on odd columns for wavefront alignment.
- Weights: `w1 = W[row][2p]`, `w2 = W[row][2p+1]` into `pe_packed_pair`.
- Skew fix: high lane from current-cycle packed product; low lane from previous-cycle packed product (`packed_prod_prev`).
- Output mapping: `results[2p] ← accum_a[bottom][p]`, `results[2p+1] ← accum_b[bottom][p]`.
- **Even `ARRAY_SIZE` only** — odd sizes `$fatal` in RTL (`tb_pe_array_packed_odd_guard.sv`).

## WP487 signed correction

Naive `[15:0]` / `[33:18]` lane extract fails signed corners; high lane needs `+1` when low lane is negative (40/125 failures → 0). Proven in isolated pair sim (`pe_packed_pair_sim.json`, 8133 vectors).

## What is proven in iverilog sim (bit-exact vs baseline `pe_array`)

| Gate | Artifact | Result |
|------|----------|--------|
| Isolated MAC pair | `bench/results/pe_packed_pair_sim.json` | PASS, 8133 vectors |
| Green array GEMM | `bench/results/pe_array_packed_sim.json` | PASS, 402 GEMMs (8×8 + 16×16) |
| Hardened shape matrix | `bench/results/pe_array_packed_hardened.json` | PASS, 25 cases, 0 constraints |
| Cycle characterization | `bench/results/packed_array_cycle_compare.json` | Identical first/full capture cycles baseline vs packed (8×8 and 16×16) |
| Top wrapper smoke | `bench/results/top_packed_smoke.json` | PASS, full datapath (compute + requant) 8×8 and 16×16 |

### Hardened shape classes (all PASS)

1. **Odd logical N** — final column pairs with `w2=0` (N=7@8, N=15@16).
2. **Rectangular GEMM** — K/M/N zero-padded to square array.
3. **32×32 tile** — corner + 16 random seeds.
4. **Batched activations** — B=2 and B=4 (baseline streaming supports `batch_count`).

### Cycle headline (iverilog-sim cycles, not silicon)

| Shape | First result | Full matrix | Packed delta |
|-------|-------------|-------------|--------------|
| 8×8 | cycle 9 | cycle 23 | 0 |
| 16×16 | cycle 17 | cycle 47 | 0 |

No throughput regression from skew/delay columns at the streaming schedule used by `pe_controller`.

## Synth-target wrapper

`rtl/top/top_packed.sv` — `pe_controller_packed` + `quantizer_array` datapath slice (not full `top.sv` UART/BRAM/FSM). Smoke-tested vs baseline `pe_controller` + `quantizer_array`.

## What is gated on Vivado (not proven here)

| Claim | Status |
|-------|--------|
| DSP count ~32 (8×8 packed) / ~128 (16×16 packed) | **UNCONFIRMED** — run `scripts/synth_packed_dsp.tcl` |
| Baseline 16×16 INT8 ~256 DSP (likely exceeds 240) | **UNCONFIRMED** |
| WNS ≥ 0 on packed configs | **UNCONFIRMED** — see `docs/evidence/packed_dsp_vivado_run.md` |

Artifact schema: `bench/results/packed_dsp_synth.json` (matches `p4_2_vivado_reports.json`).

## Four-level framing

| Level | Packed-DSP status |
|-------|-------------------|
| Simulated | Pair, array, hardened matrix, cycles, top smoke — **done** |
| CI-validated | Runners wired into `make test-host` / CI with iverilog skip — **done** |
| Synthesized | TCL + writer prepared; user must run Vivado — **pending** |
| Hardware-executed | Not in scope | 

## Regen commands

```bash
make sim-packed-pair
make sim-packed-array
make sim-packed-array-hardened
make sim-packed-cycles
make sim-packed-top
python firmware/host/write_packed_dsp_synth_json.py   # after Vivado reports exist
```
