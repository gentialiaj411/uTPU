# improvement_plan.md — uTPU Hardware Path: Road to the Full Ceiling

> Execution plan for an implementation agent. Written 2026-07-12.
> North-star context: `ceiling.md`. Status docs: `NEXT_TASK.md`, `PROJECT_STATE.md`.
> This doc supersedes the "Recommended next action" block in `NEXT_TASK.md` for
> hardware-path work; the CI-hardening task there remains valid as a parallel track.

---

## 0. How to use this document (READ FIRST, every session)

- **One phase per session.** Complete the phase, produce the phase report, STOP for
  human review. Do not start the next phase without explicit approval.
- Phases are ordered by dependency. Do not reorder. Do not skip acceptance gates.
- If a gate cannot be met, report the failure honestly and stop. Do not weaken a
  gate, relax a tolerance, or hand-edit an artifact to get green.
- If anything in this doc contradicts the repo's current state, report the
  contradiction; do not silently resolve it.

### Non-negotiable ground rules (apply to every phase)

1. **Every number regenerates from a named script.** Each phase emits a JSON
   artifact under `bench/results/` produced by a `firmware/host/run_*.py` script,
   with a matching `test_*.py` schema-lock test. Hand-edited artifacts are a
   fireable offense in this repo (it has happened before and was caught).
2. **Bit-exactness is the currency.** Any RTL change must prove: (a) ON-path
   ISA↔RTL bitmatch on the relevant vectors, (b) OFF-path / legacy programs
   byte-identical to golden fixtures, (c) no-X on output finalize paths.
3. **Deterministic metrics only.** Cycle counts from RTL counters, resource/timing
   numbers from Vivado reports, accuracy under the deployed integer contract,
   count-based stats. **Never** GPU wall-clock %, never TOPS-vs-competitor.
4. **Claim tiers.** Every result is labeled one of: `simulated (ISA-sim)`,
   `RTL-validated (iverilog)`, `synthesized (Vivado, timing-closed)`,
   `hardware-executed (on-silicon, captured)`. Never upgrade a tier implicitly.
5. **Workload-named Amdahl (standing rule — 2026-08-03).** Any Amdahl fraction,
   speedup ceiling, or “occupancy target” must name (a) the **workload** and
   (b) the **harness** it was measured on. **Multi-layer inference is the
   reference workload unless explicitly stated otherwise.** Do not grade a lever
   against an isolated-GEMM or synthetic-occupancy profile and then cite the
   number as an end-to-end inference claim. This rule exists because it was
   violated three times: `pe_occupancy` excluding finalize, `result_fetch` from
   an isolated-GEMM harness, and the Part A **2.96×** ceiling applied as if it
   were multi-layer. Reference artifacts: `cycle_attribution_mnist.json`
   (≤1.104× Part A on fused MNIST) vs `cycle_attribution.json` (≤2.96×
   isolated GEMM only).
6. **Git hygiene.** Stage only the files the phase touches. Before stopping, run
   `git diff --cached --stat` and include it in the report. NEVER stage or push
   local-only truth docs: `CLAIMS_MATRIX.md`, `RESUME_CLAIMS.md`, `context/*`,
   `PROJECT_STATE.md`, `NEXT_TASK.md`, `docs/EVIDENCE.md`, this file's claim
   updates if marked local. Do NOT push without review.
7. **Time discipline.** No single command over ~15 minutes. If a full sweep is
   intractable, run a bounded subset and validate the methodology (bounded
   bit-exact subset + fast integer-oracle full run — the established pattern).
8. **Board-gated phases (6+) must not be attempted before the physical board
   arrives (~mid-August 2026, at CMU).** Everything before that is
   sim/synthesis-verifiable on this machine (Vivado is available locally —
   see `bench/results/baseline_8x8_current_rtl_synth.json`, 2026-06-18).
9. **Demo-fit constraint.** Prefer programs that fit shipping `PROG_DEPTH=65536`
   (UART-fillable up to 65535). Larger shapes need the deferred 3-byte length
   bundle below.
   and run through the UART replay harness (`tb_uart_replay.sv` path), so the
   August board bring-up stays plug-and-play.

---

## 1. Target end state (what all of this builds toward)

A programmable INT8 inference accelerator on Artix-7 A7-100T:

- **Matrix unit:** DSP-packed 16×16 INT8 weight-stationary systolic array
  (2 MACs/DSP48, target ≈128 of 240 DSPs), timing-closed at the highest
  achievable clock (target ~100 MHz — *fenced until the synth report exists*).
- **Vector unit:** integer softmax, max-pooling, LayerNorm, GELU in RTL —
  enough for a transformer block (and later GPT-2-class decode) fully on-chip.
- **Memory hierarchy:** DDR3 → MIG → DMA → double-buffered on-chip buffers →
  array, so real-sized layers stream instead of fitting on-chip.
- **Workloads:** quantized CNNs (ResNet-class), a full transformer block, and
  ultimately a small LM (GPT-2-small 124M / SmolLM-135M-Instruct class) doing
  live text generation at an estimated ~5 tok/s (bandwidth-bound; *fenced until
  measured on-chip*).
- **Verification spine:** bit-exact host reference → ISA sim → RTL → silicon,
  at every stage.

Current verified baseline (2026-06 sessions, see `ceiling.md` §3): conv2d via
im2col + real CIFAR-10 CNN (INT8 69.11% vs float 70.45%), attention GEMMs as a
host-hybrid, per-layer + per-channel requant, quantizer X-bug fixed, current-RTL
8×8 INT8 synthesized + timing-closed at 50 MHz with **192/240 DSPs** (the
quantizer array is the hog and the critical path), pre-board UART round-trip
harness validated. On-silicon execution (P0) is OPEN.

---

## 2. Phase roadmap

| Phase | What | Gate to start | Tier when done |
|---|---|---|---|
| 0 | Baseline re-verification | none | — |
| 1 | `quantizer_array` per-lane refactor + re-synth | Phase 0 green | synthesized |
| 2 | Double-buffered load/compute overlap | Phase 1 accepted | RTL-validated + synthesized |
| 3 | Vector unit v1: integer softmax + max-pool | Phase 2 accepted | RTL-validated |
| 4 | Transformer block fully on-accelerator (sim) | Phase 3 accepted | RTL-validated (representative) |
| 5 | 16×16 INT8 via DSP packing + re-synth | Phases 1–2 accepted | synthesized |
| 6 | **BOARD-GATED:** on-silicon execution (P0) | physical board | **hardware-executed** |
| 7 | DDR3/MIG/DMA streaming memory hierarchy | Phase 6 accepted | hardware-executed |
| 8 | Small-LM decode (GPT-2/SmolLM class) + LayerNorm/GELU | Phase 7 accepted | hardware-executed |
| P | (parallel, anytime) CI-hardening per `NEXT_TASK.md` | none | CI-validated |

---

## 3. Phases in detail

### Phase 0 — Baseline re-verification (half a session)

**Goal:** prove the tree is in the state the docs claim before touching RTL.

Tasks:
1. `make test-host` — record pass/fail and any failing test names.
2. Run the ISA↔RTL bitmatch suites (fused, residual, batched, per-channel
   requant vectors) under iverilog. All must pass.
3. Confirm `bench/results/baseline_8x8_current_rtl_synth.json` exists and note
   its headline numbers (DSP 192/240, WNS +0.416 ns @ 50 MHz, LUT 25,221) as
   the before-state for Phase 1.
4. Regenerate golden fixtures list: identify the exact test set that pins
   OFF-path byte-identity (these are the gates every later phase must keep green).

**Acceptance:** all suites green (or every failure explained + human-approved as
pre-existing). **Report:** suite results, fixture inventory, baseline numbers.

---

### Phase 1 — `quantizer_array` per-lane refactor + re-synth

**Why:** the quantizer array instantiates `ARRAY_SIZE²` quantizers (~128 of the
192 DSPs at 8×8) and its finalize is the critical path pinning the design at
50 MHz. Per-lane (one quantizer per output column, `ARRAY_SIZE` total) drops
8×8 to ~80 DSPs, should raise Fmax, and is the prerequisite for 16×16 INT8
fitting in 240 DSPs at all.

Tasks:
1. Refactor `rtl/quantizer/quantizer_array.sv` from `ARRAY_SIZE²` to
   `ARRAY_SIZE` per-lane quantizers. Channels map to output columns, so
   **per-channel requant capability must be preserved** (per-output-channel
   multiplier/shift vectors, one per lane).
2. Update `rtl/top/top.sv` wiring accordingly. No other RTL changes.
3. iverilog gates (ALL before any synth):
   - existing requant ON-path vectors pass unchanged;
   - requant OFF-path byte-identical to golden fixtures;
   - no-X on finalize (rerun the check that validated the X-bug fix);
   - full ISA↔RTL bitmatch suite green (fused / residual / batched / per-channel).
4. Vivado re-synth at the current 8×8 INT8 config (`top`, `PROG_DEPTH=8192`,
   `BUFFER_SIZE=4096`, `EXT_ADDR_EN=1`, `MAX_BATCH_COUNT=4`), same flow as
   `baseline_8x8_current_rtl_synth.json`. Then attempt shorter clock periods;
   report the best period that closes (the removed path was critical, so
   expect > 50 MHz).
5. Emit `bench/results/quantizer_perlane_synth.json` (script-regenerable:
   `firmware/host/run_quantizer_perlane_synth_report.py` parsing the Vivado
   reports) with before/after DSP/LUT/FF/BRAM/WNS/achieved-clock. Add
   `firmware/host/test_quantizer_perlane_synth.py` schema-lock.

**Acceptance gates:** all four iverilog gates green; synth places + routes clean,
WNS ≥ 0; DSP at 8×8 ≤ ~112 (expected ~80); artifact regenerates from script.
**Forbidden:** claiming anything about 16×16 (that is Phase 5); touching the
scheduler, buffers, or ISA in this phase.
**Report:** per-lane structure summary, gate results, before/after table,
best closing clock, artifact path, `git diff --cached --stat`.

---

### Phase 2 — Double-buffered load/compute overlap

**Why (original):** multi-tile `compute_span_duty_cycle` is 3–5% because inter-tile
`LOAD_STATE` refills serialize against compute. Ping-pong buffering was expected
to attack that.

**Profile redirect (2026-08-03):** `bench/results/cycle_attribution.json` shows
weight `LOAD` is ~1% of `total_program_cycles` while instruction-stream cost
(fetch_decode + store/bstore + ext_addr) is ~66% on 64×64 B=48. **PE-level
concurrent LOAD+COMPUTE is parked — do not implement next.** Approach A
(compiler reorder + PE shadow) stays landed, bit-exact, default-off, and
documented cycle-neutral. It becomes useful only after the descriptor tile
sequencer (`docs/descriptor_isa_part_a_design.md`) removes the fetch/bstore tax
and LOAD actually shows up in the profile.

**Approach locked (2026-08-03):** compiler reorder (Approach A), not FSM
peek-ahead. Flag `weight_overlap` default-off; RTL `WEIGHT_OVERLAP_EN` default 0.
Ordering contract: `docs/weight_overlap_ordering.md`. Multi-tile pin:
`firmware/host/fixtures/b_multitile_weight_overlap_program_bytes.json`. B=1
legacy pin unchanged.

Tasks:
1. ~~PE shadow + `WEIGHT_OVERLAP_EN` (load_en→shadow only; `weight_commit` swap).~~
2. ~~Lowering emits overlapped schedule when `weight_overlap=True` (default off).~~
3. Mirror cycle accounting in `isa_simulator.py` for overlap (functional shadow/commit landed; cycle model TBD).
4. ~~Measure before/after with `total_program_cycles` + wall-clock ns~~ (`double_buffer_overlap.json`: bit-exact PASS, **wall-clock Δ≈0** under sequential FSM — flag stays default-off).
5. ~~Emit `bench/results/double_buffer_overlap.json` + regen script + schema test.~~
6. ~~RTL state-group attribution~~ → `cycle_attribution.json` (hypothesis confirmed).
7. **PARKED:** concurrent LOAD during COMPUTE. Do not retry until descriptor ISA
   Part A is implemented and a new attribution shows `load` as a material fraction.
8. **ORDERING LOCKED (2026-08-03): BSTORE write-arm widen first, then Part A.**
   Pre-widen rationale (frozen): `6523 = 5197 bstore + 709 compute + 617 removable`;
   Part A alone ≤1.104×; sketched 8× bstore ⇒ ~3.30× e2e.
   **Post-widen measured (2026-08-04):** total **3445**, bstore **2119** (~1.63
   cyc/word, ~2.45× on arm, **~1.89× e2e**). Part A ceiling now **≤1.218×** on
   residual. BSTORE landed at `BSTORE_WIDTH=8`.
9. **Cheap PROG_DEPTH BRAM upsizing + 64k root-cause (`prog_depth_sweep.json`):**
   First 65536 close was **invalid**: `PROG_DEPTH[15:0]` truncates 65536→0 so
   every upload is rejected and synth prunes the datapath — not a BRAM capacity
   or PC_WIDTH timing limit. RTL fix (`UPLOAD_LEN_MAX`, width-safe wr_addr)
   re-swept: **65536 closes** (WNS=+2.789, BRAM=49, RAMB36=32, LUT=41631);
   131072 also closes (protocol still caps fill at 65535). Board-fit
   `artix_a7100t_bram_max@65536` = **10/14** (256×256 / FC1-class fit).
   BUFFER_SIZE 65536 closes (37 BRAM36). Functional regression:
   `make sim-iverilog-prog-depth` / `bench/results/prog_depth_smoke.json`.
   **Shipping default `top.sv::PROG_DEPTH=65536`** (not 131072).
10. **BSTORE write-arm widen LANDED (`BSTORE_WIDTH=8`).** OOC LUT deltas
    W2/W4/W8 = +3/+10/+23 vs W1 (`bstore_widen_lut_estimate.json`); recommend 8
    against ~21.8k free LUT @65536. Functional: `bstore_wide_smoke.json` 16-word
    burst bit-exact at **1.6875 cyc/word** (pre-widen baseline 4.0). Wired
    `make sim-iverilog-bstore-wide`. Next: regen fused-MNIST attribution, then
    Part A.
11. **Buffer-resident weights (design only):** still valuable (smaller instr
    BRAM, control-only images, UART wall-clock). FC1 control 29057 fits 32768;
    restores Part A capacity justification. Parked for order, not invalid.
12. **DDR / external weight sourcing — deferred, not cancelled.** Trigger:
    models whose working set exceeds on-chip instr BRAM + unified buffer.
13. **DEFERRED BUNDLE (not shipping): 3-byte UART upload length + PROG_DEPTH
    131072 + baud increase.** Synth already closes 131072 (BRAM=81, WNS=+3.416)
    but the two-byte length field cannot fill past 65535 — so 131072 adds
    ~32 RAMB36 for **zero additional fillable shapes** vs 65536. Bundle unlocks
    the two ~106k board-fit shapes (**12/14**) and makes a 212 KB upload take
    **&lt;1 s at 3 Mbaud** instead of ~18 s at 115200. **Trigger:** a target
    model needs a program image **> 65535 words**. Until then keep shipping
    at 65536.
**MAX_BATCH_COUNT LUT bisect (independent, 2026-08-03):** @ shipping 20 ns / `PIPE_DEPTH=3`, mb24/32/48 close (LUT 32688/37293/51751); mb64 fails LUT 67217>63400. Largest fit+close = **48**. Occupancy ceiling `B/(2N+B)` at **N=8 synth**: 0.667@32, **0.750@48**, 0.800@64 unreachable. Attribution path is **N=16 → 0.600@B=48**. Label path+N; do not bare-cite either.

**TODO (cheap cycle win, do not implement in Phase 2):** requant pipe fill is
re-armed every column (`+3` cycles/column at `PIPE_DEPTH=3`). Streaming the fill
across a finalize burst would cut B=32 from +96 to ~+3. Tracked in
`bench/results/quantizer_pipe_depth_cost.json::todo_stream_requant_pipe_fill`.

**Acceptance gates:** bit-exact outputs on all measured cases; OFF-mode
byte-identical to legacy B=1 fixture; multi-tile wall-clock (`total_program_cycles`
/ Fmax) improves when overlap is on; iverilog bitmatch suite still green; re-synth
at 8×8 confirms WNS ≥ 0 after overlap RTL lands.
**Forbidden:** claiming absolute utilization numbers without the before/after
in the same artifact; touching DRAM/DMA (Phase 7); peek-ahead microarch;
**implementing concurrent LOAD+COMPUTE before descriptor profile revisit.**
**Report:** wall-clock before/after table (`total_program_cycles` primary), span
metrics secondary, cycle-count deltas, gate results, staged diff stat.

---

### Phase 3 — Vector unit v1: integer softmax + max-pooling

**Why:** the only remaining *coverage* gap that is genuinely hardware. Closes
"softmax on host" and unlocks Phase 4 (transformer block on-accelerator) and
standard conv+pool CNN topologies.

Tasks:
1. **Design doc first** (`docs/vector_unit.md`, ~1 page): chosen integer softmax
   approximation (shift/LUT exp, sum, reciprocal — cite the approach), bit
   widths, error bound, ISA encoding (new opcode(s), default-off,
   legacy-byte-identity preserved), dataflow position (post-accumulator, beside
   the quantizer). Stop for human review of the design doc BEFORE writing RTL
   if the ISA encoding requires more than one new opcode.
2. Golden model in `isa_simulator.py` + `latency_analysis.py` opcode allowlist
   update (the allowlist-completeness test will fail otherwise — that is by
   design).
3. RTL: `rtl/vector/` softmax unit + max-pool (pooling is comparators — cheap);
   wire into `top.sv` behind a parameter.
4. iverilog testbench comparing RTL vs ISA-sim golden on adversarial vectors
   (saturating, alternating, zero, random — the established distribution set),
   bit-exact.
5. **Accuracy artifact:** integer-softmax vs float-softmax divergence measured
   on real attention tensors from the existing hybrid path
   (`utpu_attention_hybrid.json` inputs). Report max/mean output delta and the
   effect on a downstream argmax/accuracy metric. This number is a finding,
   not a gate — report it whatever it is.
6. Emit `bench/results/vector_unit_softmax.json` + regen script + schema test.

**Acceptance gates:** RTL bit-exact vs ISA-sim golden on all vector sets; no-X;
legacy programs byte-identical with the unit disabled; opcode allowlist test
updated and green; `make test-host` green.
**Forbidden:** LayerNorm/GELU in this phase (Phase 8 unless trivially cheap —
ask first); any "attention on uTPU" claim (that is Phase 4's, and only at its
honest tier).
**Report:** approximation design summary, bit-exact results, accuracy-delta
finding, staged diff stat.

---

### Phase 4 — Transformer block fully on-accelerator (simulation tier)

**Why:** converts "attention GEMMs on accelerator, softmax on host" into
"transformer block runs on the accelerator (sim/RTL tier)" — a claim upgrade
the current wording explicitly forbids until this exists.

Tasks:
1. Lower the existing tested single-block transformer pattern end-to-end:
   GEMMs on the array (existing dynamic×dynamic batched-matmul path), softmax
   on the Phase 3 unit, residual via `acc_add`. If LayerNorm is still
   host-side, the block is "softmax on-chip, norm on host" — label it exactly
   that; do not round up.
2. Bit-exact chain: host integer reference → ISA sim → RTL on representative
   shapes (full block in ISA sim; RTL on the largest program that fits the
   ~12k-word iverilog threshold — the established FC1-style honest skip).
3. The demo program must fit `PROG_DEPTH=8192` and pass the UART replay
   harness — this becomes a primary August board demo.
4. Emit `bench/results/utpu_transformer_block.json` (+ script + schema test)
   recording exactly which ops ran where (`op_placement` map), bitmatch
   results, and program word counts.

**Acceptance gates:** bit-exact on the full chain at the documented scopes;
`op_placement` in the artifact is truthful (an auditor reading only the JSON
must be able to reconstruct the host/accelerator boundary); UART-harness pass.
**Report:** placement map, bitmatch table, program sizes vs PROG_DEPTH,
staged diff stat.

---

### Phase 5 — 16×16 INT8 via DSP packing + re-synth (stretch; requires 1–2 done)

**Why:** 4× the MACs of the current strong datapath. 256 INT8 MACs do not fit
240 DSPs unpacked; packing two INT8 MACs per DSP48 (the standard
INT8-packing technique) targets ≈128 DSPs for the array.

Tasks:
1. Implement DSP48 dual-INT8 packing in `rtl/PEArray/pe.sv` (parameter-gated;
   unpacked mode remains default until proven).
2. iverilog: packed vs unpacked bit-exact on the full bitmatch suite.
3. Vivado synth at 16×16 INT8 packed (with the Phase 1 per-lane quantizer,
   16 lanes): report DSP/LUT/WNS/best clock. If it does not close timing or
   fit, report the honest numbers and stop — a documented negative result is
   acceptable; a weakened gate is not.
4. Emit `bench/results/int8_16x16_packed_synth.json` + script + schema test.

**Acceptance gates:** packed==unpacked bit-exact; synth report exists (whether
it closes or not — the artifact records reality).
**Forbidden:** writing "16×16 INT8 on FPGA" anywhere unless WNS ≥ 0 and the
`.bit` generates. The board demo config remains 8×8 INT8 unless this closes.
**Report:** packing scheme, bit-exact results, resource/timing table.

---

### Phase 6 — BOARD-GATED: on-silicon execution (P0) — the category-changer

**Do not attempt before the physical board is in hand (~mid-August, CMU).**
Pre-board work is already COMPLETE (UART replay harness, `uart_replay.py`,
`run_uart_preboard_demo.py`). Confirm the board model on arrival (A7-100T
assumed; if A7-35T, 8×8 INT8 still fits, 16×16 INT4 does not — re-check
configs before flashing).

Tasks:
1. Flash the 8×8 INT8 bitstream (strong datapath; DSP budget rules out 16×16
   INT8 unless Phase 5 closed). Bring-up: clock, reset, UART link.
2. Drive the proven UART upload/capture path with the existing demo programs
   (start with the 136-word 8×8 INT8 demo, then the largest fitting MLP/CNN
   tile, then the Phase 4 transformer block program).
3. Byte-match every captured output against the ISA simulator. Capture the
   on-chip cycle counter per program.
4. Stage evidence under `docs/evidence/fpga_onboard/` (bitstream ref, Vivado
   log, program binaries, raw UART captures, host scripts, `RUN_LOG.md`,
   `REPRODUCE.md`). Add the `fpga-onboard-replay` pytest: ISA-sim byte-match
   vs the captured UART output, skipping cleanly when captures are absent.
5. Flip `on_silicon.status` to `"hardware"` in the relevant artifacts and
   populate `onchip_counter_cycles`. If any capture disagrees with the sim,
   that is the finding — record it, do not massage it.

**Acceptance gates:** ≥1 program captured byte-exact end-to-end on silicon;
evidence bundle complete; replay test green; every artifact field traceable to
a raw capture file.
**Report:** per-program capture results, cycle counts, evidence inventory.

---

### Phase 7 — Memory hierarchy: DDR3 → MIG → DMA → double-buffered tiles

**Why:** the wall between "runs a tile" and "runs a named model end-to-end."
The Arty's 256 MB DDR3 holds ~250M params at INT8; the unbuilt subsystem is
the gate. Phase 2's banking is the drop-in landing zone.

Tasks (multiple sessions; sub-phase and stop-for-review at each):
1. MIG bring-up + memory test (write/readback patterns, on board).
2. Simple DMA engine: DDR3 → buffer bank fills, driven by new ISA
   stream-descriptor ops (default-off, legacy-byte-identity preserved; golden
   model in `isa_simulator.py` first, as always).
3. Streamed-weights GEMM: a layer too large for on-chip buffers runs by
   streaming weight tiles, bit-exact vs ISA sim.
4. Scale up: ResNet-18-class CNN end-to-end with streamed weights, bounded
   bit-exact subset + fast integer-oracle full accuracy (the established
   methodology). Measured on-chip cycle counts per layer.
5. Artifact per sub-phase under `bench/results/` + scripts + schema tests.

**Acceptance gates:** per sub-phase; always bit-exact + deterministic cycles +
honest scope notes. Bandwidth utilization vs the DDR3 theoretical peak is a
reported finding.

---

### Phase 8 — Small-LM decode (GPT-2-small / SmolLM-135M-Instruct class)

**Why:** the capstone demo — live text generation off the accelerator.

Tasks (multiple sessions):
1. Vector unit v2: integer LayerNorm + GELU (same discipline as Phase 3:
   design doc → golden model → RTL → bit-exact → accuracy-delta artifact).
2. KV-cache management in DDR3 (layout + DMA descriptors).
3. Compiler: lower the decoder block; host keeps tokenizer, embedding lookup
   (or stream embeddings from DDR3), sampling, and the generate loop. Record
   the placement map truthfully.
4. Accuracy artifact FIRST, demo SECOND: perplexity (or task-accuracy) of the
   INT8 + integer-approximation model vs the float model on a fixed eval set.
   This is the make-or-break finding; if quality collapses, that is a real,
   reportable result and the demo claim gets scoped accordingly.
5. End-to-end demo: prompt in, tokens out, measured tokens/s from on-chip
   cycle counts + wall-clock of the full loop (wall-clock is acceptable here
   because it is the user-facing demo metric, clearly labeled, not a
   comparative benchmark). Estimate going in: ~4–6 tok/s, bandwidth-bound.
   Report the measured number, whatever it is.
6. Artifact: `bench/results/llm_decode_onboard.json` + script + schema test +
   a capture (text transcript + raw UART/cycle logs) under `docs/evidence/`.

**Acceptance gates:** bit-exact block-level chain (host ref → ISA sim → RTL
representative → silicon spot-checks); perplexity artifact exists before any
demo claim; placement map truthful.
**Forbidden:** "runs an LLM" without the size/speed/placement fences; any
training/fine-tuning claim, ever (forward-only datapath — this is an
inference accelerator).

---

### Parallel track P — CI-hardening (anytime, software-only)

Per `NEXT_TASK.md` "Recommended next action": wire the captured wins
(`test_utpu_conv2d_lowering.py`, `test_real_cnn_accelerator.py`,
`test_blocked_fc_requant.py`, `test_utpu_batched_matmul_lowering.py`,
`test_uart_replay.py`) into `.github/workflows/ci.yml` + `make test-host`,
fast-subset for the CNN, clean iverilog skips. As new phases land, add their
tests the same way. Upgrades claims from "locally reproduced" to "CI-validated."

---

## 4. Metric fences (targets vs claims)

| Number in this plan | Status today | Claimable when |
|---|---|---|
| ~80 DSP @ 8×8 per-lane | estimate | Phase 1 synth report |
| >50 MHz (target ~100) | estimate | Phase 1/5 timing report; write the closed number |
| ≥2× multi-tile duty cycle | estimate (expect 3–6×) | Phase 2 artifact |
| 16×16 INT8 in ≤240 DSPs | estimate | Phase 5 WNS ≥ 0 + `.bit` |
| "transformer block on accelerator" | forbidden wording | Phase 4 (sim tier) / Phase 6 (silicon tier) |
| byte-exact on silicon | not yet true | Phase 6 capture |
| ResNet-class end-to-end streamed | not yet true | Phase 7 artifact |
| ~5 tok/s small-LM decode | bandwidth estimate | Phase 8 measured |
| INT8 within ~1.3% of float (LM) | unknown | Phase 8 perplexity artifact |

## 5. Key file pointers

- Backend op registry: `firmware/host/graph_passes.py` (`_BACKEND_SUPPORTED_OPS`)
- uTPU lowerings: `firmware/host/lowering_*.py`, `utpu_conv2d_lowering.py`,
  `utpu_batched_matmul_lowering.py`
- ISA + golden sim: `firmware/host/isa_encoder.py`, `isa_simulator.py`,
  `latency_analysis.py` (opcode allowlist)
- Requant: `firmware/host/requantization.py`; RTL `rtl/quantizer/*`
- Array: `rtl/PEArray/pe.sv`, `pe_array.sv`; top: `rtl/top/top.sv`;
  buffers: `rtl/unified_buffer/`; UART: `rtl/UART/*`
- UART preboard harness: `firmware/host/uart_replay.py`,
  `run_uart_preboard_demo.py`, `rtl/tb/tb_uart_replay.sv`
- Synth baseline: `bench/results/baseline_8x8_current_rtl_synth.json`
- Board configs: `firmware/host/board_config.py`

## 6. Phase report template (use verbatim, end of every phase)

```
PHASE N REPORT
1. What changed (files, one line each)
2. Gate results (each acceptance gate: PASS/FAIL + the number)
3. Artifacts (path + regen command for each)
4. Findings / surprises (honest, including negative results)
5. Ambiguities / TODO-VERIFY
6. git diff --cached --stat output
STOPPED FOR REVIEW.
```
