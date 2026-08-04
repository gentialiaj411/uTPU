# Hardware design space (Artix A7-100T)

_Generated 2026-08-04T17:37:06.628575+00:00 · git `7169b29` · sweep_status=`partial`_

## Scope

Full-top Vivado synth/impl (route, no bitstream) on `xc7a100tcsg324-1` at `PROG_DEPTH=65536`, `QUANTIZER_PIPE_DEPTH=3`, `BUFFER_SIZE=4096`.

Grid: `ARRAY_SIZE ∈ {4,8}` × `COMPUTE_DATA_WIDTH ∈ {4,8}` × `MAX_BATCH_COUNT ∈ {4,16,48}` × `period ∈ {20,15,12,10}` ns, plus a `16×16 INT4` Pareto attempt.

## Occupancy (peak ≠ achieved)

- **occupancy** = `0.205806`
- **occupancy_source** = steady_state_attribution.json
- cold_compute_share_until_steady_bitexact
- **Throughput** is computed once per `(ARRAY_SIZE, CDW, MAX_BATCH_COUNT)` from that config's **tightest closing period** (see `throughput_by_config`). Per-period rows below are timing evidence only.
- **Fmax derivation**: `achieved_fmax_mhz_from_period_minus_wns = 1000 / (clock_period_ns - WNS_ns)`. This understates capability at loose constraints: Vivado stops optimizing once the applied period is met, so loose-period (period−WNS) Fmax values are floors, not measurements of the design's true speed. Throughput (peak/achieved GOP/s) is therefore taken from each configuration's tightest closing period only.
- **peak_gops** = `ARRAY_SIZE^2 * 2 * Fmax_GHz`
- **achieved_gops** = `peak_gops * occupancy` (never report peak as achieved)
- **margin_class**: `comfortable` (>1 ns), `thin` (0.2-1 ns), `marginal` (<0.2 ns).

## Shipping point rationale

Chosen shipping point: **N=8 INT8 MAX_BATCH_COUNT=48 @ 12.0 ns (~83.3 MHz)** (prefix `dss_n8_cdw8_mb48_clk12_pd3_prog65536`), margin_class=`thin`.

Why:
1. **Accuracy** — INT8 per-layer accuracy 97.33% vs INT4 58.32% (`real_model_accelerator.json`).
2. **Board fit** — closes on xc7a100t with PROG_DEPTH=65536 instruction BRAM and remaining LUT/DSP/BRAM headroom for the MNIST/FC class.
3. **Batch ceiling** — MAX_BATCH_COUNT=48 is the largest closing batch from the timing-closure / LUT bisect path (mb64 LUT-oversubscribes).
4. **Clock** — **12 ns (~83 MHz)** is the shipping default (WNS thin but non-marginal). Loose-period closes are floors under met constraints; **100 MHz is the demonstrated ceiling**, not the shipping default.

Evidence: WNS=0.271 ns (thin), constraint Fmax=83.33 MHz, period-WNS Fmax≈85.2587603376247 MHz, LUT=45885/63400, DSP=72/240, BRAM=49/135.
Config throughput (from tightest close @10.0 ns, WNS=0.012): peak=12.815378454144975 GOP/s, achieved=2.6374755657442055 GOP/s.
Highest util resource at shipping close: **lut** (not oversubscribed).

## Demonstrated 100 MHz ceiling

- Constraint **100 MHz** (10 ns) closed with **WNS=0.012 ns** (`marginal`).
- Any claim citing 100 MHz must carry the WNS value inline (WNS=0.012 ns, margin_class=marginal). Shipping default remains 12 ns / ~83 MHz.

## Accuracy–throughput Pareto

- Point A (INT8): accuracy=97.33%, achieved_gops=2.6374755657442055 (tightest close @10.0 ns, WNS=0.012)
- Point B (16x16 INT4): **did not close** or not yet attempted — see not_yet_attempted / timing table.

## Throughput by config (one row per NxCDWxMB)

| N | CDW | MB | source_period_ns | WNS | margin | Fmax_MHz | peak_GOP/s | achieved_GOP/s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 8 | 8 | 4 | 20.0 | 4.636 | comfortable | 65.08721687060661 | 8.331163759437645 | 1.714599450055527 |
| 8 | 8 | 16 | 10.0 | 0.16 | marginal | 101.6260162601626 | 13.008130081300813 | 2.6771449136842604 |
| 8 | 8 | 48 | 10.0 | 0.012 | marginal | 100.12014417300762 | 12.815378454144975 | 2.6374755657442055 |

## Timing evidence by corner (per period)

| N | CDW | MB | period_ns | status | WNS | margin | Fmax_MHz | LUT | DSP | BRAM | binder |
|---:|---:|---:|---:|---|---:|---|---:|---|---|---|---|
| 8 | 8 | 4 | 20.0 | closed | 4.636 | comfortable | 65.08721687060661 | 16594/63400 | 72/240 | 49/135 | bram |
| 8 | 8 | 16 | 20.0 | closed | 3.285 | comfortable | 59.82650314089142 | 26237/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 16 | 15.0 | closed | 1.099 | comfortable | 71.93727069994965 | 26228/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 16 | 12.0 | closed | 0.992 | thin | 90.84302325581396 | 26239/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 16 | 10.0 | closed | 0.16 | marginal | 101.6260162601626 | 26239/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 48 | 20.0 | closed | 2.789 | comfortable | 58.10237638719424 | 41631/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 48 | 15.0 | closed | 1.017 | comfortable | 71.5154115711936 | 45885/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 48 | 12.0 | closed | 0.271 | thin | 85.2587603376247 | 45885/63400 | 72/240 | 49/135 | lut |
| 8 | 8 | 48 | 10.0 | closed | 0.012 | marginal | 100.12014417300762 | 46021/63400 | 72/240 | 49/135 | lut |

## Summary counts

- sweep_status: **partial**
- closed (attempted): **9**
- failed (attempted): **0**
- not_yet_attempted: **43**

## Not yet attempted

| N | CDW | MB | period_ns | canonical_prefix |
|---:|---:|---:|---:|---|
| 4 | 4 | 48 | 20.0 | `dss_n4_cdw4_mb48_clk20_pd3_prog65536` |
| 4 | 4 | 48 | 15.0 | `dss_n4_cdw4_mb48_clk15_pd3_prog65536` |
| 4 | 4 | 48 | 12.0 | `dss_n4_cdw4_mb48_clk12_pd3_prog65536` |
| 4 | 4 | 48 | 10.0 | `dss_n4_cdw4_mb48_clk10_pd3_prog65536` |
| 4 | 4 | 16 | 20.0 | `dss_n4_cdw4_mb16_clk20_pd3_prog65536` |
| 4 | 4 | 16 | 15.0 | `dss_n4_cdw4_mb16_clk15_pd3_prog65536` |
| 4 | 4 | 16 | 12.0 | `dss_n4_cdw4_mb16_clk12_pd3_prog65536` |
| 4 | 4 | 16 | 10.0 | `dss_n4_cdw4_mb16_clk10_pd3_prog65536` |
| 4 | 4 | 4 | 20.0 | `dss_n4_cdw4_mb4_clk20_pd3_prog65536` |
| 4 | 4 | 4 | 15.0 | `dss_n4_cdw4_mb4_clk15_pd3_prog65536` |
| 4 | 4 | 4 | 12.0 | `dss_n4_cdw4_mb4_clk12_pd3_prog65536` |
| 4 | 4 | 4 | 10.0 | `dss_n4_cdw4_mb4_clk10_pd3_prog65536` |
| 4 | 8 | 48 | 20.0 | `dss_n4_cdw8_mb48_clk20_pd3_prog65536` |
| 4 | 8 | 48 | 15.0 | `dss_n4_cdw8_mb48_clk15_pd3_prog65536` |
| 4 | 8 | 48 | 12.0 | `dss_n4_cdw8_mb48_clk12_pd3_prog65536` |
| 4 | 8 | 48 | 10.0 | `dss_n4_cdw8_mb48_clk10_pd3_prog65536` |
| 4 | 8 | 16 | 20.0 | `dss_n4_cdw8_mb16_clk20_pd3_prog65536` |
| 4 | 8 | 16 | 15.0 | `dss_n4_cdw8_mb16_clk15_pd3_prog65536` |
| 4 | 8 | 16 | 12.0 | `dss_n4_cdw8_mb16_clk12_pd3_prog65536` |
| 4 | 8 | 16 | 10.0 | `dss_n4_cdw8_mb16_clk10_pd3_prog65536` |
| 4 | 8 | 4 | 20.0 | `dss_n4_cdw8_mb4_clk20_pd3_prog65536` |
| 4 | 8 | 4 | 15.0 | `dss_n4_cdw8_mb4_clk15_pd3_prog65536` |
| 4 | 8 | 4 | 12.0 | `dss_n4_cdw8_mb4_clk12_pd3_prog65536` |
| 4 | 8 | 4 | 10.0 | `dss_n4_cdw8_mb4_clk10_pd3_prog65536` |
| 8 | 4 | 48 | 20.0 | `dss_n8_cdw4_mb48_clk20_pd3_prog65536` |
| 8 | 4 | 48 | 15.0 | `dss_n8_cdw4_mb48_clk15_pd3_prog65536` |
| 8 | 4 | 48 | 12.0 | `dss_n8_cdw4_mb48_clk12_pd3_prog65536` |
| 8 | 4 | 48 | 10.0 | `dss_n8_cdw4_mb48_clk10_pd3_prog65536` |
| 8 | 4 | 16 | 20.0 | `dss_n8_cdw4_mb16_clk20_pd3_prog65536` |
| 8 | 4 | 16 | 15.0 | `dss_n8_cdw4_mb16_clk15_pd3_prog65536` |
| 8 | 4 | 16 | 12.0 | `dss_n8_cdw4_mb16_clk12_pd3_prog65536` |
| 8 | 4 | 16 | 10.0 | `dss_n8_cdw4_mb16_clk10_pd3_prog65536` |
| 8 | 4 | 4 | 20.0 | `dss_n8_cdw4_mb4_clk20_pd3_prog65536` |
| 8 | 4 | 4 | 15.0 | `dss_n8_cdw4_mb4_clk15_pd3_prog65536` |
| 8 | 4 | 4 | 12.0 | `dss_n8_cdw4_mb4_clk12_pd3_prog65536` |
| 8 | 4 | 4 | 10.0 | `dss_n8_cdw4_mb4_clk10_pd3_prog65536` |
| 8 | 8 | 4 | 15.0 | `dss_n8_cdw8_mb4_clk15_pd3_prog65536` |
| 8 | 8 | 4 | 12.0 | `dss_n8_cdw8_mb4_clk12_pd3_prog65536` |
| 8 | 8 | 4 | 10.0 | `dss_n8_cdw8_mb4_clk10_pd3_prog65536` |
| 16 | 4 | 48 | 20.0 | `dss_n16_cdw4_mb48_clk20_pd3_prog65536` |
| 16 | 4 | 48 | 15.0 | `dss_n16_cdw4_mb48_clk15_pd3_prog65536` |
| 16 | 4 | 48 | 12.0 | `dss_n16_cdw4_mb48_clk12_pd3_prog65536` |
| 16 | 4 | 48 | 10.0 | `dss_n16_cdw4_mb48_clk10_pd3_prog65536` |

## Artifacts

- JSON: `bench/results/design_space_sweep.json`
- Plot: `docs/design_space_roofline.png`
- Runner: `firmware/host/run_design_space_sweep.py`
- TCL: `scripts/synth_design_space_point.tcl`

