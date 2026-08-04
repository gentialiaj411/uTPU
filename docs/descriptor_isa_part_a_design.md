# Part A — Descriptor ISA + Hardware Tile Sequencer (design proposal)

**Status:** encoding **FROZEN** (2026-08-03). Golden model landed
(`firmware/host/tile_gemm_golden.py` — written from this spec, not from
lowering). Host checks 1–2 pass; RTL sequencer check 3 still open.
**Trigger evidence:** `bench/results/cycle_attribution.json` (isolated GEMM) +
`bench/results/cycle_attribution_mnist.json` (multi-layer reference).

---

## 0. Dual justification — capacity vs performance (keep separate)

### 0a. Capacity — upload-length width bug at 64k; buffer-resident weights restore Part A

Evidence: `bench/results/program_word_composition.json` +
`bench/results/prog_depth_sweep.json` + `bench/results/board_fit_audit.json` +
`docs/buffer_resident_weights_design.md`.

| Program | Words | Embedded payload | Control | Payload % |
|---|---:|---:|---:|---:|
| Fused MNIST case1 (BSTORE path) | 1486 | **1296** | 190 | **87.2%** |
| Capstone FC1 256×196 (legacy STORE) | 43201 | **14144** | **29057** | 32.7% |
| Board-fit 256×256 | 53137 | **17408** | **35729** | 32.8% |

#### Cheap `PROG_DEPTH` upsizing (Artix, mb48/pd3 @ 20 ns)

Prior framing assumed Pynq-Z2 BRAM tiers on an Artix part with ~114 free BRAM36.

| PROG_DEPTH | Status (post-fix) | BRAM | RAMB36 | WNS | Fetch on critical? |
|---:|---|---:|---:|---:|---|
| 8192 | closed | 21 | 4 | +2.691 | no |
| 16384 | closed | 25 | 8 | +2.200 | **yes** |
| 32768 | closed | 33 | 16 | +2.292 | no |
| **65536** | **closed** (after fix) | **49** | **32** (≥29) | **+2.789** | no |
| 131072 | closed (after fix) | 81 | 64 | +3.416 | no — UART length still caps at 65535 |

**Root cause of the first bad 64k close (not “instr BRAM mapping collapse”):**
`UPLOAD_LEN_HI_STATE` compared length to `PROG_DEPTH[15:0]`. At
`PROG_DEPTH=65536` (=`0x10000`) that slice is **0**, so every nonzero upload is
rejected → FSM never reaches `UPLOAD_BODY` / fetch-decode → synth constant-folds
the datapath (`Synth 8-3333` on `prog_len` / `upload_count` / FSM onehot;
post-route RAMB36 collapsed to 2, LUT~11k). Same width family as the original
131072 hard error (`upload_count[PC_WIDTH-1:0]` OOR at PC_WIDTH=17).

RTL fix: `UPLOAD_LEN_MAX = min(PROG_DEPTH, 65535)` and width-safe
`bram_wr_addr <= upload_count`. FC1 (43201) and 256×256 (53137) are both
**< 65535**, so the two-byte length field is not a blocker for those shapes.

`artix_a7100t_bram_max@65536` board-fit: see regenerated
`bench/results/board_fit_audit.json` (FC1 and 256×256 fit as full in-image
programs). Protocol note: uploads still cannot exceed 65535 words without
widening the length field — `PROG_DEPTH=131072` BRAM is synthesizable but not
fully fillable via the current UART header.

#### Buffer-resident weights (design option — not implemented)

`BUFFER_SIZE=65536` already closes (37 BRAM36, bank-integral, LUT-as-Memory=0).
If weights are filled into the unified buffer via a dedicated host path
(`docs/buffer_resident_weights_design.md`) instead of embedded `BSTORE`/`STORE`
payload:

| Shape | Control-only image | Fits |
|---|---:|---|
| FC1 | 29057 | **32768 already** (also fits full in-image @65536 after fix) |
| 256×256 | 35729 | Part A compression **or** full in-image @65536 (53137 closes) |

Post-fix, full in-image FC1/256×256 fit `PROG_DEPTH=65536` without buffer-fill.
Buffer-resident weights remain the better long-term shape (smaller instr BRAM,
no multi-second UART program images, Part A capacity justification on
control-only images).

**Part A capacity justification returns under this option.** Part A was not
wrong; it was insufficient *alone* while payload remained in the program image.
With control-only images, compressing control is exactly what admits large
shapes. Keep Part A **parked for implementation** (BSTORE write-arm first) but
**do not document it as capacity-invalid**.

DDR/external sourcing stays deferred to models that exceed on-chip instr+buffer
budget (ResNet-scale), not cancelled.

#### Performance (unchanged)

Part A alone on fused MNIST remains ~**1.10×** (or ~**1.45×** post-BSTORE).
BSTORE write-arm stays the primary **cycle** lever (79.7%) and is independent of
the capacity options above.

### 0b. Performance (secondary; ~1.1× on multi-layer today)

On the **reference workload** (fused MNIST multi-layer, fast-UART TB, N=16):

| Group | Cycles | % |
|---|---:|---:|
| `bstore` | 5197 | **79.67%** |
| `compute` | 709 | 10.87% |
| all other | 617 | 9.46% |

If Part A removes the non-bstore/non-compute remainder and **bstore survives**:

`ceiling = 6523 / (5197 + 709) = 6523 / 5906 =` **≤ 1.104×**

After an 8× BSTORE write-arm speedup first (ordering locked in
`improvement_plan.md`): residual ≈ 1976 cycles, Part A's 617 become ~31% ⇒
Part A then yields ~**1.45×** (compounds). Grade sequencer claims against the
**named multi-layer** ceiling, not isolated-GEMM 2.96×.

### 0c. Why still not concurrent LOAD+COMPUTE

Isolated-GEMM attribution (`cycle_attribution.json`, 64×64 B=48, N=16): `load`
is **1.1%**. PE-level double-buffering cannot move the multi-layer profile either
until a post-descriptor (and preferably post-bstore-fix) attribution shows
material `load` share. Approach A stays parked.

> Naming note: earlier handoff text called this “Phase 5 descriptor ISA.”
> In `improvement_plan.md` today, Phase 5 is 16×16 packing and stream-descriptor
> ops appear under Phase 7. Treat this document as the **Part A design gate**
> for the descriptor/sequencer work regardless of phase number renumbering.

---

## 0d. Dual ARRAY_SIZE — do not conflate

| Path | N | dtype | Streaming ceiling @ B=48 |
|---|---:|---|---:|
| Artix-7 **synth/impl** | **8** | INT8 | **0.750** (`timing_closure_sweep.json`) |
| RTL **sim / attribution / MNIST** | **16** | INT4 | **0.600** (`cycle_attribution.json`) |

Never write bare `0.750` or `0.600` without path+N.

---

## 1. Opcode widening (3 → 4 bits) — FROZEN

Widen opcode space with a **4-bit low nibble** for new ops. Legacy decode remains
`instruction & 0x7` for all existing programs.

**Compatibility / decode rule (FROZEN):** recognize `TILE_GEMM` only when the low
nibble is **exactly** `0x8`. Do **not** treat `instruction[3]==1` as “new opcode”
— legacy FETCH/LOAD/RUN/NOP already use bit[3] as flags (low nibbles `0x9` /
`0xA` / `0xB` / `0xD`). New ops require `descriptor_isa=True` /
`DESCRIPTOR_ISA_EN=1` and compiler flag `emit_tile_descriptors=False` by default.

| Encoding (low nibble) | Mnemonic | Notes |
|---|---|---|
| `0x0`–`0x7` | legacy ops | Decoded via `word & 0x7` (flags may set bit[3]) |
| `0x8` | `TILE_GEMM` | Only safely unique new opcode in v1 |
| `0x9`–`0xF` | reserved | Several collide with legacy flag encodings; do not assign yet |

**Instruction word stays 16-bit.** Descriptor payloads use following words.

---

## 2. `TILE_GEMM` descriptor encoding — FROZEN

One descriptor replaces the per-tile micro-op sequence:
`LOAD weights → LOAD inputs → RUN accumulate → … → RUN finalize`
(payload host→buffer via existing `BSTORE`/`STORE` remains outside the descriptor).

### Word 0 — opcode header (FROZEN)
```
[15:10] batch_m1 = batch_size - 1     // 6 bits → B in [1, 64]; shipping ceiling B=48
[9:6]   flags:
          bit0: acc_clear on first K-tile of each out-block (ib==0)
          bit1: finalize (quantize writeback, no ReLU in v1) after each out-block's last K-tile
          bit2: hoist_payloads_already_in_buffer (addresses use hoist layout)
          bit3: weight_overlap_commit (PARKED — ignored unless WEIGHT_OVERLAP_EN)
[5:4]   reserved = 0   // do not steal for ReLU yet without a fixture bump
[3:0]   opcode = TILE_GEMM (0x8)
```

`finalize_last` means quantize+writeback only in golden v1 (`relu=False`), matching the
blocked-FC INT4 path used in attribution (`apply_relu=False`). ReLU-as-a-flag is a
deliberate follow-up if needed — not part of this freeze.
### Words 1–5 — addresses + geometry (FROZEN)
| Word | Field |
|---|---|
| 1 | `weight_base` (low `address_width` bits) |
| 2 | `input_base` |
| 3 | `result_base` |
| 4 | `out_blocks[15:8] \| in_blocks[7:0]` |
| 5 | `array_size[15:8] \| dtype_code[7:0]` (`dtype_code` = compute_data_width: 4 or 8) |

**Footprint:** **6 instruction words** per full GEMM (all tiles). Sequencer walks
`out_blocks × in_blocks` on-chip.

**Hoist layout** (when flag bit2 set), matching `lowering_blocked_fc_utpu.py`:
- weight tile `(ob,ib)` at `weight_base + (ob*in_blocks+ib) * (N*N/items_per_word)`
- input tile `ib` at `input_base + ib * (N*B/items_per_word)`
- result out-block `ob` at `result_base + ob * (N*B/items_per_word)` (INT4 packing)

---

## 2b. Golden model (independent oracle) — landed host-side

**Must be written from this frozen spec**, not refactored out of
`lowering_blocked_fc_utpu.py`. A lowering-derived “golden” inherits whatever the
emitter already does and cannot catch spec bugs.

Implementation: `firmware/host/tile_gemm_golden.py` (NumPy; no lowering import).
Fixture pin: `firmware/host/fixtures/tile_gemm_frozen_encoding.json`.
Tests: `firmware/host/test_descriptor_isa_tile_gemm.py`.

Three-way checks:

| # | Check | Status |
|---|---|---|
| 1 | golden vs fully-unrolled microcoded emission (same GEMM → same FETCH bytes) | **pass** (catches spec bugs) |
| 2 | golden vs ISA-simulator `TILE_GEMM` expansion | **pass** |
| 3 | golden vs RTL sequencer | **open** (sequencer not started; needed regardless of BSTORE vs Part A ordering) |

---

## 3. Sequencer FSM state cluster (design; RTL not started)

```
DESC_LATCH_STATE          // decode TILE_GEMM, latch descriptor fields
DESC_TILE_SETUP_STATE     // compute current (oi, ii) addresses
DESC_LOAD_W_STATE         // buffer → PE weights (reuse LOAD_STATE datapath)
DESC_LOAD_X_STATE         // buffer → PE inputs / batch chunk
DESC_COMPUTE_STATE        // reuse COMPUTE_STATE / pe_controller
DESC_NEXT_TILE_STATE      // advance ii/oi; loop or fall through
DESC_FINALIZE_STATE       // reuse COMPUTE_WRITEBACK + requant pipe
DESC_DONE_STATE           // return to FETCH_BRAM for next host instruction
```

**Parked overlap hook:** concurrent LOAD+COMPUTE only after a post-descriptor
profile shows material `load` share. Do not build concurrency in Part A.

---

## 4. Analytical program-size reduction (`board_fit_audit.json`)

Source shapes: `bench/results/board_fit_audit.json::per_shape` (**ARRAY_SIZE=16**,
same N as the attribution grid).

### 4a. Upper bound (if entire program were control microcode)

> **Caveat (2026-08-03):** these ratios assume the program is *control*. Real
> programs embed weight/activation payload (`program_word_composition.json`).
> Fused MNIST is 87% payload → Part A word shrink ~1.14×, not 36×+. Treat the
> table as a **control-only** upper bound, not a capacity claim.

| tag | shape | today words | descriptor control | ratio |
|---|---|---:|---:|---:|
| demo_smallest | 16×16 | 217 | 6 | **36.2×** (control-only fiction) |
| demo_tiny | 16×32 | 424 | 6 | 70.7× |
| demo_2x2_blocks | 32×32 | 847 | 6 | 141.2× |
| first_overflow_pynqz2_baseline | 32×64 | 1675 | 6 | 279.2× |
| (64×64 from audit) | 64×64 | 3349 | 6 | 558.2× |

### 4b. Isolated-GEMM cycle mix (harness only — not the grading yardstick)

On hoisted 64×64 B=48 (`program_words=5029`, **N=16**, fast-UART TB):
- instruction stream = **66.2%**
- `result_fetch` = 26.7% — **harness artifact** on multi-layer (§4d)
- `load` = 1.1%

Isolated-GEMM Part A Amdahl (strip instruction stream): **≤ 2.96×**.
Post-cut residual on that harness ≈ 79% drain / ~0.17 compute — a **harness
floor**, not a multi-layer estimate. Keep for archaeology; **do not grade
Part A against it.**

### 4c. Multi-layer Amdahl (reference grading ceiling) — §0b detail

From `cycle_attribution_mnist.json` (fused MNIST case1, fast-UART, N=16):

```
total            = 6523
bstore           = 5197   (79.67%)
compute          =  709   (10.87%)
removable (else) =  617   ( 9.46%)
ceiling_x        = 6523 / 5906 = 1.104470…  →  cite ≤1.104×  (~1.10×)
```

**Grade the RTL sequencer’s cycle claims against ≤1.104× on this workload**,
not against 2.96×. Capacity acceptance remains word-count reduction (§5).

### 4d. `result_fetch` harness artifact + BSTORE path (next-lever measurement)

**Drain:** fused MNIST `result_fetch` = **24 cycles (0.37%)** vs isolated GEMM
**6144 (26.7%)**. Inter-layer activations stay in the unified buffer; UART drain
is what a single-GEMM harness does. Verdict:
`result_fetch_harness_artifact_confirmed`.

**BSTORE write-arm (landed `BSTORE_WIDTH=8`):** pre-widen was 79.67% of
multi-layer cycles at **4.0 cyc/word**. Post-widen fused MNIST:
`bstore=2119` (61.5%), total `3445` (~**1.89×** e2e). Smoke:
`bstore_wide_smoke.json` @ **1.6875 cyc/word**. Artifacts:
`run_bstore_path_measure.py`, `bstore_path_measure.json`,
`cycle_attribution_mnist.json`.

| Quantity | Pre-widen | Post-widen (W=8) |
|---|---:|---:|
| Payload words (case1) | 1296 / 13 bursts | same |
| `perf_attr_bstore` | 5197 | 2119 |
| Cycles / payload word | **4.0** | **~1.63** |
| Program total | 6523 | 3445 |

FSM: skid-fill `BSTORE_FETCH_DATA` → wide `BSTORE_WRITE` (up to 8 banks/beat).
`unified_buffer.done` remains registered. Pre-widen identity `1296×4+13=5197`
is frozen in `bstore_path_measure.json::measured`.

**Buffer write width** (`unified_buffer.sv`):

| Config | BANKS | Compute-port write | BSTORE store-port |
|---|---:|---|---|
| N=16 INT4 (sim/attr) | **64** | 64×16-bit / cycle | up to **8**×16-bit / beat |
| N=8 INT8 (synth) | **32** | 32×16-bit / cycle | same (`STORE_WIDE=8`) |

OOC LUT deltas vs W1: +3 / +10 / +23 for W2/W4/W8 (`bstore_widen_lut_estimate.json`).
Skid + BRAM latency prevent a full 8× on the arm (~2.45× measured).

**Ordering:** BSTORE landed. Part A remains the next sequencer lever (capacity +
post-widen ≤1.218× Amdahl on residual). Golden check 3 proceeds for correctness.

---

## 5. Part A acceptance

1. Flag default-off; legacy B=1 fixture byte-identical.
2. Spec-derived golden checks 1–2 pass; check 3 (RTL) deferred until weight-source
   rescope is decided (sequencer not started).
3. Artifact claims attach UART-baud / on-chip-core-cycle scope **and** N path
   **and** named workload (multi-layer inference is the reference unless stated).
4. **Capacity:** with in-image weights, Artix 32k helps mid shapes (9/14) but
   not FC1/256×256. Root cause of the bad 64k close was `PROG_DEPTH[15:0]`
   truncation (fixed; re-sweep pending). Buffer-resident weights restore Part A
   capacity justification (§0a, `docs/buffer_resident_weights_design.md`) —
   Part A is parked for *implementation order*, not because capacity is invalid.
5. **Cycle claims (if any):** grade against multi-layer **≤1.104×** today, or the
   post-BSTORE residual (~1.45× Part A after 8× bstore) — named workload required.
6. No concurrent LOAD+COMPUTE in Part A.
7. No BSTORE pipeline/widen in the Part A change (BSTORE is ordered **first**,
   separate change).

## 6. Explicit non-goals (Part A)

- DDR3/MIG/DMA (Phase 7).
- Widening ARRAY_SIZE / INT8 packing (separate Phase 5 stretch).
- Enabling `weight_overlap` by default.
- Claiming board wall-clock without a 115200 measurement.
- Treating harness `result_fetch` dominance or isolated-GEMM 2.96× as
  multi-layer inference facts.
- Implementing BSTORE widen/pipeline in the same change as the sequencer.
