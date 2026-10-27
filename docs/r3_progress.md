# R3 progress

## Chunk 1 - GEMV

- Built `rtl/decode/gemv_unit.sv`: registered 16-lane INT4/INT8 stream GEMV shell,
  nibble unpack, INT32 accumulation, start/done and weight valid/ready handshakes.
- Runner: `python tools/decode/run_gemv_rtl.py`; planned cases are 224 real vectors
  and 2,000 random/edge vectors. Artifact records `not_run` because `iverilog` is
  absent in this environment; no RTL mismatch claim is made.
- Open issue: execute the runner with iverilog and complete exact scale/requant
  wiring before sign-off.

## Chunk 2 - vector units

- Added registered start/done shells for act quant, RMSNorm, RoPE, SiLU/mul, and
  residual units in `rtl/decode/vector_units.sv`.
- Runner emits 2,000 real + 2,000 random planned cases per unit. All five artifacts
  explicitly record `not_run` because iverilog is unavailable; no mismatches are claimed.

## Chunk 3 - attention

- Added parameterized KV-cache, score, softmax, and attention-value registered
  interfaces in `rtl/decode/attention_units.sv`.
- Attention artifact records the required model/position/KV-bit matrix as `not_run`
  because no RTL simulator is installed; exact score/KV differential remains open.

## Chunk 4 - classifier/argmax

- Added registered classifier-chunk and tied embedding-row interfaces.
- Runner artifact records the required 200-token/20-full-logit plan as `not_run`
  because iverilog is unavailable.

## Chunk 5 - sequencer and top

- Added 4 KiB stream reader, microcoded-sequencer shell, parameterized `top_decode`,
  and `tools/decode/assemble.py`.
- Top artifact records 0.5/1/2-byte-per-cycle throttle cases as `not_run`; RTL
  simulation is blocked by missing iverilog.

## Chunk 6 - end to end and gate

- Added `run_decode_rtl.py`, fail-closed `decode_gate.py`, and structural checker
  enforcing 50 MHz/no divide/no initial rules.
- E2E and gate are not passed: iverilog is unavailable and all RTL runner artifacts
  remain explicitly `not_run`; gate therefore exits non-zero as designed.

## U1 GEMV - DONE (redone by coordinator, 2026-10-26)

- `rtl/decode/gemv_unit.sv`: 16 MAC lanes + one shared 6-stage scaler
  (acc*act_scale*rshift_rne(mult,16) -> rshift_rne(.,shift-16) -> div127 via Q32 reciprocal
  + one-step correction -> accumulate). Chunks shorter than 16 beats back-pressure.
- Golden: `tools/decode/golden_gemv.py` wraps `runq_runtime._linear_high` (stories260K, 3 tokens),
  plus 300 random tiles (random multipliers/shifts/act scales, k=64 and k=172, INT8/INT4) and
  40 directed RNE-tie tiles.
- `python tools/decode/golden_gemv.py && python tools/decode/run_gemv_rtl.py`:
  666 real tiles / 10,536 elements and 340 random+directed tiles / 5,440 elements, 0 mismatches.
  Mutation build (row 3 LSB flip): 1,006/1,006 tiles caught.
- Fault-injection check of the harness: ties-half-up 640 mismatches, no div127 correction 563,
  dropped final-beat product 11,876.
- Vivado 2025.2 OOC place+route, xc7s50, 20 ns: WNS +1.870 ns, 4,953 LUT, 2,499 FF, 25 DSP, 0 BRAM.
  Scaler uses full 64-bit multiplies; narrow if DSP budget tightens.
- Measured 101.3 cycles/tile in sim (includes start/drain); INT8 streams 16 B/beat, INT4 8 B/beat.

## U2 activation quantization - DONE (redone by coordinator, 2026-10-26)

- Agent version (c2a1617) rejected: reciprocal was a 32-step divider unrolled into one cycle (Vivado
  synthesis failed; the multi-cycle DIV state was dead code); "5 mismatches" fault counts were the
  capped MISMATCH-line count; ACTQ_BAD_TIE also forced code+1, so a pure tie fault went undetected
  (0 mismatches); the real/random split was hard-coded.
- `rtl/decode/act_quant.sv`: peak tracked during capture, 31-cycle restoring divider with div_rne
  fixup for recip = RNE(127<<24 / peak), then a 2-stage multiply / rshift_rne(.,24) / clamp pipeline.
- Golden wraps `runq_runtime._quantize` (3 stories260K tokens) plus directed groups: exact RNE ties
  (peak 254*2^j makes recip 2^(23-j)), +-127 clamp, peaks near 2^31, zero/single-nonzero, k=172 tail,
  and 300 random full-range groups.
- `python tools/decode/golden_act_quant.py && python tools/decode/run_act_quant_rtl.py`:
  333 real groups / 7,572 elements and 337 directed+random groups / 21,548 elements, 0 mismatches
  (codes and scales). Mutation: 670/670 groups caught. 120.9 cycles/group.
- Fault injection (source-text edits, full mismatch counts): ties-half-up 766, reciprocal truncation
  8,292, peak off-by-one 1,298, clamp at 128 98.
- Vivado 2025.2 OOC place+route, xc7s50, 20 ns: WNS +6.791 ns, 427 LUT, 202 FF, 4 DSP, 1 RAMB18.

## U3 RMSNorm - DONE (coordinator, 2026-10-26)

- `rtl/decode/rmsnorm.sv` (+ `decode_pkg.sv`, `seq_mul.sv`): capture with registered x*x into int64,
  then mean = div_const_rne(ss, D) + eps (fixed.py small/wide-path rounding and the small path's
  sequential correction reproduced exactly), 47-cycle isqrt(mean<<30), 55-cycle div_rne(2^54, root),
  and a sequential per-element chain (one radix-16 seq_mul, ~55 cycles/element) with int64 wrap.
- Golden `tools/decode/golden_rms.py` wraps `_rms_high`: 33 real stories260K vectors plus 250
  directed/random (D=64 and D=288, div ties, x*recip ties via eps-forced root 2^31/2^32,
  gamma/output-shift ties, zero vectors). 46,112 elements, 0 mismatches; mutation 46,112/46,112.
- Fault injection (`tools/decode/fault_inject.py rms`): 5 detected; 2 proven equivalent
  (div_rne(2^54, root) can never tie; a half-up tie estimate is corrected back to RNE).
- OOC xc7s50 @ 50 MHz: WNS +1.432 ns, 2,611 LUT, 961 FF, 12 DSP, 1 BRAM36.

## U4+U5 vec_alu (RoPE, SwiGLU, residual add, embedding scale) - DONE (coordinator, 2026-10-26)

- `rtl/decode/vec_alu.sv`: one sequential element unit, one shared 33x33 DSP multiplier (3 cycles per
  product). Ops: ROPE (_rope_high, cos/sin ROMs from `tools/decode/gen_luts.py`), SWIGLU
  (_hidden_high(_silu_high(h1), h3), sigmoid ROM, int32 wrap of silu, low-64 product via two 32x16
  partial products), ADD (int32 wrap), MULSHIFT (tied-embedding row scaling).
- Golden `tools/decode/golden_vec.py` wraps _rope_high/_silu_high/_hidden_high/_embedding_high
  (48 real calls) plus 240 directed/random cases: RoPE exact ties (x0*c = 2^14 mod 2^15), sigmoid
  index edges +-512/513, silu and output ties via tie-forcing configs, add wrap-around, embedding ties.
  12,372 beats, 0 mismatches; mutation 12,372/12,372.
- Fault injection: 6/6 detected (rope sign, pair-index wrap, sigmoid edge, dropped high product word,
  ties-away rounding, saturating add).
- OOC xc7s50 @ 50 MHz: WNS +2.834 ns, 3,006 LUT, 561 FF, 4 DSP, 3 BRAM36 (h8 RoPE + sigmoid ROMs).

## U2 update - qmax 7 (2026-10-26)

- `act_quant` gained `qmax7` (KV4 cache quantization, `_quantize(k, hs, 7)`). Golden now also wraps a
  kv_bits=4 runtime: 666 real groups + 377 directed (incl. exact ties at qmax 7), 0 mismatches.
  Fault injection 5/5 detected (adds qmax-7 clamp and numerator faults).

## U6 attention - DONE (coordinator, 2026-10-26)

- `rtl/decode/attention.sv` (+ `mul_pipe.sv` 64x33 pipelined multiplier, `div_const_pipe.sv`
  pipelined div_const_rne for d=7/127 with shift-subtract residual): on-chip KV cache (codes in
  CB=8 or packed CB=4 bits + 32-bit scale per head), per head: 8-lane dot -> 3 wide multiplies ->
  rshift_rne -> div127 -> div127|div7 scores; softmax via exp ROM and restoring divider; 16-position
  AV blocks with _to_base, block peak, reciprocal, INT8 codes, prob.code, rshift_rne(.*peak,15), div127.
- Golden `tools/decode/golden_attn.py` wraps RunQIntegerLlama._score and _attention_value over 40
  decode steps (positions 0-63) for KV8 and KV4: 3,200 real head calls; plus 420 random-cache calls
  (incl. 2^27 V scales that reach the -128 code clamp). 28,960 outputs, 0 mismatches (KV8 on CB=8,
  KV4 on packed CB=4); mutation 28,960/28,960.
- Fault injection: 6 detected, 1 proven equivalent (exp_q15 is 0 from index 710, so a 1023/1024
  clip is unobservable).
- OOC xc7s50 @ 50 MHz: WNS +2.336 ns, 4,602 LUT, 27 DSP, 11.5 BRAM36 (260K config).

## U7-U9 classifier, sequencer, top, end to end - DONE (coordinator, 2026-10-26)

- Classifier = GEMV over the tied INT4 table + argmax (strictly-greater, lowest index on ties, the
  runtime's lexsort order), inside `top_decode`.
- `rtl/decode/top_decode.sv`: fixed-function sequencer FSM (no microcode ROM; the agent's
  `assemble.py` stub and its unchecked structural test were removed), 8 INT32 scratch buffers,
  4 KiB prefetch FIFO fed by a segment prefetcher (embedding codes, embedding scales, layers,
  classifier) over an 8-bit valid/ready flash port with seek, 1 GEMV beat/cycle.
- Image: `tools/decode/pack_decode_image.py` (16-row tiles, per-group scales before codes;
  classifier tiles double as the tied embedding table). stories260K 237,632 B; stories15M
  13,073,472 B (fits the 14,155,776 B flash budget).
- End to end: `tools/decode/run_decode_rtl.py` compares every logit and the argmax of every step
  with RunQIntegerLlama.forward (greedy, prompt then generated tokens). Each unit's mutation build
  is caught end to end (~1,200-1,500 logit mismatches each). Results in the gate table below.
- Throughput (stories260K, 232,554 streamed bytes/token): 0.25 B/cycle 930,326 cycles/token
  (= stream bound); 0.5 B/cycle 470,071 (98.9% of bound); 1 B/cycle ~365k (the sequencer pops at
  most 1 B/cycle, so non-overlapped GEMV compute is exposed; double-buffered tiles would fix it).
- Full engine OOC xc7s50 @ 50 MHz (260K config): WNS +0.777 ns, 24,372 LUT, 32,746 FF, 73 DSP,
  19.5 BRAM36. Register-built arrays (tile buffers, scale register files) dominate FF/LUT; moving them
  to BRAM is required before the 15M configuration can fit (KV cache alone ~56 BRAM36 at L128 KV4).

## Gate status (2026-10-26, `python tools/decode_gate.py`)

| item | result |
|---|---|
| units (gemv, act_quant, rms, vec_alu, attention) | all pass, 0 mismatches, mutation caught on every element |
| fault injection | 26 detected, 4 proven equivalent, 0 unexplained |
| OOC xc7s50 @ 50 MHz | all blocks and the full 260K engine meet timing (top WNS +0.777 ns) |
| e2e stories260K KV8, 1 B/cycle | 5 prompts, 169 steps, 86,528 logits + tokens, 0 mismatches, 424,026 cycles/token |
| e2e stories260K KV4, 1 B/cycle | 5 prompts, 169 steps, 86,528 logits + tokens, 0 mismatches, 424,026 cycles/token |
| e2e stories260K KV8, 0.5 B/cycle | 2 prompts, 19 steps, 0 mismatches, 470,071 cycles/token (98.9% of stream bound) |
| e2e stories260K KV8, 0.25 B/cycle | 1 prompt, 10 steps, 0 mismatches, 930,326 cycles/token (= stream bound) |
| e2e stories15M KV4, 1 B/cycle | "Once upon a" -> "time, there was a little girl named": 11 steps, 352,000 logits + tokens, 0 mismatches, 13,287,867 cycles/token (11,633,661 streamed bytes; 87.5% of stream bound; 3.76 tok/s at 50 MHz). Simulation of the 15M configuration only: it does not fit the xc7s50 as built. |

Cycle counts are data-independent: the KV8 and KV4 runs produce different tokens but identical cycles.
Cycles/token are means over the positions run (up to 35); attention cost grows with position.
At 1 B/cycle the engine is above the 232,554-cycle stream bound because the sequencer pops at most
one byte per cycle, so GEMV compute between tiles is not hidden (double-buffered tiles would fix it).

## M1 - top_decode with 15M parameters, baseline (2026-10-26)

Config: stories15M (D 288, H 768, NL 6, NH/NKV 6, HS 48, VOCAB 32000), MAX_SEQ 128, KV4, h48 RoPE ROMs
(12,288 rows). Synthesized through a generated wrapper `build/decode/top_decode_15m_wrap.sv`
(parameters from `build/decode/decode_image_15m/params.json`). Record: `top_decode_15m` in
`bench/results/decode_synth_ooc.json`. Place failed on utilization DRC (numbers are post-synthesis):

| resource | used | xc7s50 | % |
|---|---|---|---|
| Slice LUTs | 73,601 (6,208 LUTRAM) | 32,600 | 226 |
| Slice FFs | 122,750 | 65,200 | 188 |
| DSP48 | 73 | 120 | 61 |
| BRAM36 | 112 | 75 | 149 |
| F7 mux | 18,387 | 16,300 | 113 |

Hierarchical (post-synth, `build/decode/synth/top_decode_15m/synth_util_hier.rpt`):

| instance | LUT | FF | BRAM36 | DSP |
|---|---|---|---|---|
| top_decode body (scr, tbuf, mrf/srf, qc, fifo, FSM) | 25,540 | 114,889 | 0 | 3 |
| attention (all of it) | 6,439 | 3,498 | 112 | 25 |
| gemv_unit | 35,812 | 2,507 | 0 | 25 |
| rmsnorm | 2,595 | 993 | 0 | 12 |
| vec_alu | 2,672 | 660 | 0 | 4 |
| act_quant | 543 | 203 | 0 | 4 |

Reading: the 112 BRAM36 are all KV cache (224-bit word x 4,608 entries for K and again for V; Vivado rounds
the 4,608-deep memory up to 8,192, i.e. 7 wide x 8 deep = 56 each). ~115K FFs in the top body are the
register-built `tbuf` (16 x 768 B = 98 Kbit), `mrf`/`srf`. The gemv instance at 35.8K LUT (vs 4,953 standalone)
is not explained by its own logic; to be re-measured after the register arrays move to RAM.

## M2 - 15M build fits (2026-10-27)

No numerics changed: every edit is storage mapping or read-pipeline timing; the golden is untouched.

| | before (M1, post-synth, place failed) | after (placed + routed, 50 MHz) | limit |
|---|---|---|---|
| LUT | 73,601 | 20,289 (62%) | <= 85% = 27,710 |
| FF | 122,750 | 9,223 | 65,200 |
| DSP | 73 | 73 | 120 |
| BRAM36 | 112 | 67 | <= 68 |
| WNS | n/a (not placed) | +0.525 ns | >= 0 |

Margin warning: BRAM36 is 67 of the 68 budget (75 on the part). Anything that adds a memory needs to take one out.

Per-array moves:

| array | before | after |
|---|---|---|
| KV cache (attention) | separate K and V, 224-bit x 4,608 deep each, depth rounded to 8,192: 112 BRAM36 | one merged memory (K entries [0,4608), V [4608,9216)), word split into <=72-bit slices, each made of explicit banks sized to a BRAM36 aspect (72x512 ... 9x4096), last partial bank trimmed to a power of two: 57 RAMB36 + 1 RAMB18 |
| `tbuf` (16 rows x 768 B tile buffer) | register array, ~98 Kbit of FFs | 128-bit x KBMAX BRAM with byte write enables, registered read: 4 RAMB36 |
| `mrf`/`srf` (per-row group scales) | registers | per-row LUTRAM with async read (writes gated, `g_sc`) |
| scratch (`scr`) | one array sized for the widest role | per-role offsets, `SCR_N = 6*max(D,KVD) + 2*H`: 4 RAMB36 |
| RoPE cos/sin | two 16-bit ROMs | one 32-bit `{cos,sin}` ROM, truncated to MAX_SEQ 128 (3,072 rows) |
| sigmoid, exp, `qc`, rmsnorm `xmem`/`gmem`, act_quant `mem` | BRAM | `ram_style`/`rom_style = "distributed"` (LUTRAM) |

After (hierarchical, `build/decode/synth/top_decode_15m/`): top body 2,690 LUT / 1,184 FF / 9 RAMB36 + 1 RAMB18;
attention 6,719 LUT / 3,728 FF / 57 RAMB36 + 1 RAMB18 / 25 DSP; gemv 5,126 LUT; rmsnorm 2,608; vec_alu 2,468;
act_quant 687. The 35.8K-LUT gemv figure in M1 was the register `tbuf` being absorbed into it; it is 5.1K now.
The 260K config (`top_decode_260k`, default parameters) is 15,492 LUT / 7,001 FF / 73 DSP / 12 BRAM36 / WNS +0.829.

New files: `rtl/decode/luts/rope_cs_q15_h{8,48,48_p128}.mem`. `gen_luts.py` writes them.

## M3 - re-verification on the M2 RTL (2026-10-27)

Unit differentials (all `status: pass`, 0 mismatches, mutation caught on every expected output):

| runner | mismatches | mutation caught |
|---|---|---|
| gemv (260K tiles) | 0 | 1006/1006 |
| gemv15 (new: 2,328 real stories15M tiles from `_linear_high` + 120 random, K=288 and K=768, INT8/INT4; 39,168 outputs) | 0 | 2448/2448 |
| act_quant | 0 | 1043/1043 |
| rmsnorm | 0 | 46,112/46,112 |
| vec_alu | 0 | 12,372/12,372 |
| attention (260K) | 0 | 28,960/28,960 |
| attention15 (new: HS 48, 6 layers, 6 KV heads, MAX_SEQ 128, KV4; 858 head calls, 648 real captured at positions up to 127 + 210 random; 41,184 outputs) | 0 | 41,184/41,184 |

Fault injection (`tools/decode/fault_inject.py`, `bench/results/decode_fault_injection.json`): gemv 4 detected + 1 proven equivalent;
actq 5; rms 5 + 2 equivalent; vec 7 (new: rope_cos_sin_lane_swap); attn 9 + 2 equivalent (new: kv_v_region_offset, kv_tail_bank_half,
kv_write_v_offset, kv4 ones as before; kv_bank_sel_undelayed is proven equivalent, reason in the entry); top 7 (new unit: tbuf lane swap,
embed address, scale group/row selection, scratch overlaps; each flips every logit); attn15 1 (kv_bank_write_ge).

Two test gaps found and closed, not papered over:
- `kv_bank_write_ge` (bank match `==` -> `<=`) was NOT detected by the 260K attention runner. Aliased writes land on higher
  addresses, and that runner wrote KV entries in address order, so a later correct write always repaired them. It is a real bug
  class at 15M geometry: with 512-deep banks and 768 entries per layer, a K write at position >= 43 aliases to a later layer's
  earlier position. Fixed by making the 15M attention runner write in decode order (position, then layer, then K before V);
  the mutation is then detected (41,052 mismatches). The fault lives under `attn15`, and the gate requires it.
- The first 15M GEMV golden had 3,488 tiles with empty expected values (the runtime returns only the requested row slice and
  my capture indexed absolute rows). The regenerated golden pads to absolute rows. The random-tile shift range was also narrowed
  (36..47) because most lower shifts overflow INT32 at K >= 288, which made capture take hours.

Synthesis (placed + routed, xc7s50, 50 MHz), all met: gemv_unit +1.870 ns, act_quant +5.058, rmsnorm +0.960, vec_alu +3.366,
attention +2.452 (27 DSP, 8 BRAM36), top_decode_260k +0.829 (15,492 LUT, 12 BRAM36), top_decode_15m +0.525 (20,289 LUT, 67 BRAM36).

End to end (golden = runtime logits and tokens), all 0 logit and 0 token mismatches:

| run | steps | logits compared | cycles/token |
|---|---|---|---|
| 260K KV8, 1 B/cycle, 5 prompts x 32 | 169 | 86,528 | 424,026 |
| 260K KV4, 1 B/cycle | 169 | 86,528 | 424,026 |
| 260K KV8, 0.5 B/cycle | 169 | 86,528 | 514,861 |
| 260K KV8, 0.25 B/cycle | 169 | 86,528 | 950,247 |
| 15M KV4, 1 B/cycle, 1 prompt x 8 | 11 | 352,000 | 13,287,867 (3.76 tok/s at 50 MHz) |

Note on the 0.5 / 0.25 B/cycle rows: the M3 numbers (514,861 / 950,247) cover 5 prompts and 169 steps. The older figures in the first gate table (470,071 / 930,326) cover 2 prompts / 19 steps and 1 prompt / 10 steps. Mean cycles/token grows with position, so the two sets are not comparable. It is not a regression: the coordinator reran the old 2-prompt workload on a69cc17 and got 470,070.84 cycles/token, identical to before.

`tools/decode_gate.py` now also requires: gemv15/attention15 passes, the attn15 and top fault units, `top_decode_15m` timing met,
<= 68 BRAM36 and <= 85% LUT for it, and the 15M e2e artifact. Gate output: unmet = [].

Open: BRAM margin is 1 BRAM36 (67/68). The 15M e2e covers one prompt of 8 tokens (11 steps); the full matrix ran on 260K only.

## M4 - QSPI flash controller (2026-10-27, simulation + synthesis only, nothing run on hardware)

### Flash part (source)
- Boolean schematic (Real Digital, `https://www.realdigital.org/downloads/63f9a8205ebd9c2e8c2d265ad25097dc.pdf`, sheet 10 "QSPI"):
  the flash symbol is labelled **S25FL128SAGMFI00** (Spansion/Cypress/Infineon S25FL128S, 128 Mb = 16 MB, 3.3 V, SOIC-16). Nets
  QSPI_DQ0..DQ3, QSPI_CS and QSPI_SCK (FPGA pin CCLK_0 = C8). The product page (`realdigital.org/hardware/boolean`) says only "16MB QSPI".
  The schematic text extracted from the PDF is garbled; the part name and the CCLK_0/C8 to QSPI_SCK link read cleanly, the other pin
  locations do not and were NOT used.
- Datasheet used: S25FL128S/S25FL256S rev 05 (Spansion, 12 Jul 2012; copy hosted by NXP at
  `community.nxp.com/.../S25FL128S_256S_00.pdf`). I did not find the current Infineon revision (downloads blocked), so figures below are
  from rev 05. The ordering-code suffix of the actual part (S25FL128SAGMFI00) was not decoded against the datasheet's ordering table.
- Read command: **Quad Output Read 6Bh** (3-byte address; 6Ch for 4-byte). Address and instruction on IO0 only, data 4 bits/clock on IO[3:0],
  falling-edge launch, high nibble first, auto-increment, wraps at the end of the array (sec. 10.4.4, Fig. 10.30). The 16 MB device is
  fully reachable with 24-bit addresses, so `ADDR_BYTES=3` (parameter; 4 gives 6Ch).
- Dummy cycles: **8** for 6Bh at latency codes 00/01/10 up to 104 MHz (tables 8.10/8.12; 0 dummy clocks only with LC=11 at <= 50 MHz);
  Fig. 10.30 shows 8 dummy clocks. The table text came from a PDF-to-text extraction with scrambled columns; I cross-checked it with
  Fig. 10.30 but did not read the table graphically. Default LC (CR1[7:6], non-volatile) is 00. `DUMMY` is a parameter. If someone had
  programmed LC=11 the reader would misalign (documented, not detected).
- Quad mode: **CR1[1] QUAD, non-volatile** (table 8.9); it must be 1 for 6Bh to drive IO2/IO3 ("The QUAD bit of Configuration Register must
  be set (CR Bit1=1)"). It is written with WRR (01h) of {SR1, CR1}. The reader does not write it: it assumes QUAD=1. Open item: confirm on
  the board (Vivado's flash programming for a x4 configuration normally sets it; not verified here).
- Timing used: tV <= 7.65 ns (30 pF, 2.7-3.6 V), tHO >= 2 ns, tCS(read) >= 10 ns, tCSS/tCSH >= 3 ns, QOR fmax 104 MHz.

### Clock pin (7-series)
- The flash clock is CCLK_0, reachable after configuration only through STARTUPE2 (USRCCLKO, USRCCLKTS=0). `rtl/board/qspi_pads.sv` is the thin
  wrapper (STARTUPE2 + 4 IOBUF + OBUF for CS#). It synthesizes in Vivado (1 STARTUPE2, 4 OBUFT, 1 OBUF; no errors). It has no pin
  constraints (see M5 below). The core `qspi_reader.sv` contains no primitives and simulates in iverilog.
- "First 3 USRCCLKO edges after configuration are swallowed" is taken from the task brief; I could not confirm it in an official document in
  this session. Handled in the core: after reset it sends `STARTUP_PULSES` (4 pulses = 8 toggles) on SCLK with CS# high, which the flash ignores.
  Caveat: this keeps SCLK idle-low only if the number of swallowed transitions is even (3 clock cycles lost = 6 transitions). If exactly 3
  transitions were lost the pin would idle high after startup and every command would be inverted; first bring-up should look at SCLK with a
  scope or ILA before trusting reads. Unverified on hardware.

### Design (`rtl/decode/qspi_reader.sv`)
- Interface is top_decode's flash port as is: in `seek_req, seek_addr, st_ready`; out `st_valid, st_data`. `st_valid = ob_valid && !seek_req`, so no
  byte is offered in the seek cycle. One byte register plus one nibble register is the only buffering; the 4 KiB FIFO stays in top_decode. Zero BRAM.
- SPI mode 0, SCLK = clk/(2*HALF) (HALF=1: 25 MHz at a 50 MHz core, 0.25 B per core cycle). `SAMPLE_LATE=1` (default) samples IO on the core
  cycle that generates the falling edge (about one SCLK half period after the flash launched the data, before it changes); `0` samples at the
  rising edge. Both are simulated; which one meets timing on the board depends on real pad and clock-to-out delays (open issue).
- On `seek_req` (latched, newest address wins): stop at SCLK low and, inside the instruction/address/dummy phase, at a byte boundary; raise CS#;
  hold CS# high `CS_HIGH`=2 core cycles (40 ns >= tCS 10 ns); CS# low; 8-bit opcode + 24-bit address on IO0 (IO2/IO3 driven high, IO1 input), IO
  released after the last address bit; `DUMMY` clocks; then nibbles are captured and assembled high nibble first. Bytes assembled but not yet
  delivered are dropped on a seek.
- Backpressure: SCLK is stopped low (CS# held) when the next rising edge would complete a second byte while the output register is occupied and not being taken.
- Parameters: `ADDR_BYTES, DUMMY, HALF, SAMPLE_LATE, STARTUP_PULSES, CS_HIGH`. `-DQSPI_MUTATE` swaps the nibbles.

### Verification
Flash model `tools/decode/tb/qspi_flash_model.sv` was written from the datasheet sections above, not from the reader. It `$fatal`s on: wrong
opcode, address >= image size, CS# raised mid-byte of the instruction/address/dummy phase, CS# edge while SCLK is high, CS# high time < tCS, host not
driving IO0 during instruction/address, host driving the bus while the flash drives data. It drives the data lines Hi-Z during the dummy clocks and
X from tHO to tV after each falling edge (unit test: tV 7.65 / tHO 2 ns at a 20 ns clock; e2e scales these to its 2 ns clock). The model cannot see a
wrong dummy count directly; it shows up as X/Z or shifted bytes, which the byte comparison catches.

| item | result |
|---|---|
| unit `run_qspi_rtl.py` -> `decode_qspi_rtl.json`: address 0, last byte, every alignment 0..39, random aligned/unaligned, back-to-back seeks, 250 seeks mid-burst, 60 seeks in the cycle of a byte handoff, 4000-cycle stall (0 SCLK edges), slow consumers; 5 configs (3-byte HALF1 late/early, HALF2 late, 4-byte, HALF3 early) | 0 mismatches; 15.4k-16.1k bytes per config compared with the image; 60-71 same-cycle seek/handoff events per config |
| mutation `QSPI_MUTATE` (nibble swap) | caught on exactly the expected bytes in all 5 configs (e.g. 15,082 of 15,082) |
| fault injection `qspi` | 10 detected (dummy-1, nibble order, address byte order, opcode 0Bh, CS# not raised on seek, byte valid in seek cycle, stale byte kept on seek, no backpressure stall, abort mid command byte, stale capture after seek), 2 proven equivalent (reasons in the JSON) |
| OOC `qspi_reader` @ 50 MHz | 63 LUT / 98 FF / 0 DSP / 0 BRAM, WNS +15.255 ns |
| OOC `top_decode_15m_qspi` (15M top_decode + reader, wrapper from `tools/decode/gen_qspi_wrap.py`) | 20,366 LUT (62%), 9,323 FF, 73 DSP, 67 BRAM36, WNS +1.148 ns (standalone 15M was +0.525; routing variation, the reader adds 77 LUT) |
| e2e stories260K KV8, 1 prompt, gen 8 through reader + model (`decode_e2e_rtl_kv8_p1_qspi.json`) | 10 steps, 5,120 logits + tokens, 0 mismatches, **930,673 cycles/token** |
| same workload, ideal stream 0.25 B/cycle (`decode_e2e_rtl_kv8_p4_1prompt.json`) | 0 mismatches, 930,326 cycles/token |

QSPI overhead: +347 cycles/token (+0.037%) over the ideal 0.25 B/cycle stream, about 87 cycles for each of the 4 seeks per token (abort, CS# gap,
40 command SCLK cycles = 80 core cycles). The engine is stream-bound at 0.25 B/cycle, so on 260K the flash is the bottleneck.

Stories15M through QSPI was NOT simulated (the 1 B/cycle run took about 3 hours; 0.25 B/cycle would take several times longer). Estimate: 11,633,661
streamed bytes per token x 4 cycles/byte = 46.53 M cycles, plus about 350 for seeks, so **about 1.07 tok/s at 50 MHz**. The 260K run is within 0.04% of its
stream bound at this rate, so compute should be hidden behind the slower stream. This is an estimate, not a measurement.

### Gate
`tools/decode_gate.py` now also requires: `decode_qspi_rtl` pass (all configs), the `qspi` fault unit, `qspi_reader` (0 BRAM) and `top_decode_15m_qspi`
(timing, <= 68 BRAM36, <= 85% LUT) synth entries, and the QSPI e2e artifact (flash == qspi). Gate: exit 0, unmet = [].
`tools/decode/run_decode_rtl.py` gained `--flash qspi`; the ideal path was re-run on a short workload and gives identical cycles to the previous
script. top_decode and every other unit are untouched.

### Open issues
- Nothing ran on the board: STARTUPE2 startup-edge behaviour, the QUAD bit state, the sample point (early/late) against real pad and clock-to-out
  delays, and IO2/IO3 floating/high during the instruction phase (the reader drives them high) are all unverified.
- The flash part is read from the schematic's symbol label; datasheet rev 05 was used.
- BRAM margin is still 1 BRAM36 (67/68); the reader adds none.
- M5 (board top, UART, XDC) was not started. Real Digital's master constraints file lists the clock (F14), switches, LEDs, buttons, displays, UART_rxd (V12) and
  more, but no QSPI pins, and the QSPI pin locations in the schematic extraction are unreliable. They must not be guessed; someone needs to read them off the
  schematic (bank-14 dedicated D00..D03 / FCS_B pins) before `rtl/board/qspi_pads.sv` can be constrained.

## M5 - Boolean board top (2026-10-27, simulation + full implementation; no bitstream generated, no hardware access)

### Pin verification (hard-stop check passed)
`python tools/board/pin_check.py` -> `bench/results/boolean_pin_check.json`. Each pin is checked against (a) the device database of the installed
Vivado 2025.2 (`link_design -part xc7s50csga324-1`, 324 package pins, PIN_FUNC and BANK), (b) Real Digital's master XDC where it lists the pin, and
(c) the Boolean schematic, rasterized sheets 10, 11 and 13 read by eye (the PDF text layer is scrambled, so the schematic column is a
transcription, not machine output). VCCO_0, VCCO_14 and VCCO_15 are all VCCp3V3 (sheet 13), CFGBVS_0 is tied to 3V3 (sheet 10), so LVCMOS33 everywhere.

| signal | pin | device PIN_FUNC (bank) | master XDC | schematic | agrees |
|---|---|---|---|---|---|
| clk 100 MHz | F14 | IO_L13P_T2_MRCC_15 (15) | `clk` F14 | sheet 11 net UCLK from IC9 ASEM1-100.000MHZ | yes |
| UART_rxd (FPGA RX) | V12 | IO_L24N_T3_D16_14 (14) | `UART_rxd` V12 | UART_TXD_IN | yes |
| UART_txd (FPGA TX) | U11 | IO_L24P_T3_D17_14 (14) | `UART_txd` U11 | UART_RXD_OUT | yes |
| qspi_cs_n | M13 | IO_L6P_T0_FCS_B_14 (14) | not listed | QSPI_CS to IC8 pin 7 | yes |
| qspi_dq[0] | K17 | IO_L1P_T0_D00_MOSI_14 | not listed | QSPI_DQ0 to IC8 pin 15 | yes |
| qspi_dq[1] | K18 | IO_L1N_T0_D01_DIN_14 | not listed | QSPI_DQ1 to IC8 pin 8 | yes |
| qspi_dq[2] | L14 | IO_L2P_T0_D02_14 | not listed | QSPI_DQ2 to IC8 pin 9 (WP#), 1.8k pull-up | yes |
| qspi_dq[3] | M15 | IO_L2N_T0_D03_14 | not listed | QSPI_DQ3 to IC8 pin 1 (HOLD#), 1.8k pull-up | yes |
| flash SCLK | C8 | CCLK_0 (0) | n/a | QSPI_SCK to IC8 pin 16, via STARTUPE2 | yes |

The coordinator's recalled Arty S7 pins were checked and REJECTED: L13 is `IO_0_14` (not FCS_B, which is M13) and M14 is `IO_L5P_T0_D06_14`
(not D03, which is M15). The flash is wired to the config pins (D00..D03, FCS_B), not to other pins.

### Design
- `rtl/board/boolean_top.sv`: MMCME2_BASE (100 MHz x10 / 20 = 50 MHz, BUFG on output and feedback), reset held until the MMCM is locked + 16 cycles,
  `qspi_pads` (STARTUPE2 + IOBUF/OBUF), `boolean_core`. `boolean_core.sv` (no vendor primitives): `uart_rx`/`uart_tx` (8N1, `CPB` = 434 for
  115200 baud at 50 MHz), `board_ctrl`, `top_decode` (defaults = stories15M, checked against `params.json` by `impl_top.py`) and `qspi_reader`.
  Image addresses are offset by `FLASH_BASE` = 0x280000 (2.5 MiB; the xc7s50 bitstream is 17,536,096 bits = 2,192,012 B; 0x280000 is 256 kB
  aligned; image 13,073,472 B ends at 15,694,912 < 16 MiB).
- UART protocol (little endian, `board_ctrl.sv` header): `'T' tok_lo tok_hi pos idx_lo idx_hi` -> `'R' tok_lo tok_hi cyc[4] logit[4]` (argmax token, core
  cycles start to done, INT32 logit at idx, 0 if idx >= VOCAB). `'S' ntx nrx tx...` -> `'s' nrx rx...` single-bit SPI on the flash (ntx, nrx <= 8) for
  RDID/RDSR/RDCR/WREN/WRR; `'V'` -> `'v' 01`. Full logit streaming is not offered: top_decode emits logits with no backpressure and the UART is ~4000x slower
  than the classifier, so the host probes individual logit indices instead. The host must wait for the response before the next frame.
- Host tool `tools/board/boolean_host.py` (pyserial; `selftest` checks the frame encoding against a software stub, never run against a board).

### Verification
| item | result |
|---|---|
| `tools/board/run_uart_rtl.py` -> `boolean_uart_rtl.json`: TB speaks the UART protocol to `boolean_core` (260K parameters, KV8) + flash model with the image at flash offset 0x280000 and QUAD powering up as 0 | 0 mismatches, 66 checks: V, RDID (01 20 18 4D 00 80), RDCR=00, truncated frame dropped by timeout, WREN, SR1.WEL, WRR (CR1=02), WIP polling (01, 01, 00), CR1 read back 02, then 4 decode steps (prompt 1,403,260 + 2 generated): every argmax token and one probed logit per step (one out-of-range probe expecting 0) bit-exact to the integer golden; reported cycles 930,674 per step |
| mutation `BOARD_MUTATE` (token bytes swapped in the response) | caught |
| negative: decode request with QUAD = 0 | flash model `$fatal`s "Quad Output Read issued with CR1.QUAD = 0" |
| `tools/board/impl_top.py` -> `boolean_top_impl.json`: synth + place + route of `boolean_top` with `constraints/boolean_xc7s50.xdc` at 50 MHz | 20,726 LUT (63.6%), 9,815 FF, 73 DSP, 67 BRAM36, WNS +1.418 ns, WHS +0.008 ns, TNS/THS 0; DRC: 0 errors, 37 warnings (DPIP-1 DSP input pipelining x36, IOSR-1 x1) |
| bitstream | NOT generated (`write_bitstream` is not in the implementation Tcl) |

Flash model changes for M5 (it is shared with M4, so the M4 unit test, qspi faults and the QSPI e2e were rerun on it: unit 5/5 configs pass, 10 detected + 2 equivalent
faults, e2e 10 steps 0 mismatches 930,673 cycles/token): `BASE` image offset, `QUAD_INIT`, and register commands (RDID/RDSR1/RDCR/WREN/WRR) behind `ALLOW_REG_CMDS`;
a Quad read with CR1.QUAD = 0 is fatal.

### Bring-up commands (none of these were run)
All from the repo root. Vivado = `C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat`.
1. Pack the 15M image if `build/decode/decode_image_15m/image.bin` is missing (`python tools/decode/pack_decode_image.py ...` as used for the 15M runs).
2. Bitstream + flash file: `vivado.bat -mode batch -nojournal -nolog -source tools/board/make_bitstream.tcl` (refuses to write a bitstream if WNS < 0). It writes
   `build/boolean/boolean_top.bit` and `build/boolean/boolean_flash.mcs` (bitstream at 0x0, image at 0x280000, config x1 so boot does not depend on the QUAD bit).
3. First test over JTAG with the volatile bitstream only (Vivado Hardware Manager, Program Device with `boolean_top.bit`), then find the USB-UART COM port and run
   `python tools/board/boolean_host.py --port COMx ping` (expect `board version 1`) and `... id` (expect `01 20 18 4d 00 80`: manufacturer 01, S25FL128S 2018, ID-CFI length 4D,
   byte 4 = 00 means 256 kB uniform sectors; if it reads 01 use `s25fl128sxxxxxx1-spi-x1_x2_x4` in `program_flash.tcl`).
4. Check / set the Quad-enable bit: `... cr1` prints `SR1=.. CR1=.. QUAD=n`; if QUAD=0 run `... enable-quad` (WREN, WRR of SR1 unchanged + CR1|0x02, poll WIP, read back).
   QUAD is CR1[1], non-volatile (datasheet table 8.9), so it only has to be done once; the WRR rewrites the other CR1 bits as read (the OTP bits stay as they are).
5. Program the flash: `vivado.bat -mode batch -nojournal -nolog -source tools/board/program_flash.tcl` (erase/program/verify 16 MB, several minutes), then power-cycle.
6. Run: `python tools/board/boolean_host.py --port COMx step --token 1 --pos 0 --probe 5`, then `... run --prompt 1,403,260 --gen 8` (greedy; token ids). Expected for
   stories15M at 0.25 B/cycle: about 46.5 M core cycles/token, roughly 0.9 s/token. The returned `cyc` is the board's own cycle count.

### Open issues (M5)
- Never run on hardware: STARTUPE2 startup edges and SCLK idle polarity (see M4), QUAD state, sample point against pad delays, the host tool, `make_bitstream.tcl` and
  `program_flash.tcl` (written from Vivado's standard flow and the part names in the installed `get_cfgmem_parts`, but not executed), UART at 115200 on the FT-bridge.
- Flash pin timing is unconstrained (no `set_input_delay`/`set_output_delay` for qspi_dq/qspi_cs_n: the clock reaches the pin through STARTUPE2, so a proper constraint needs a
  generated clock on USRCCLKO that I did not set up). The pad registers are not forced into the IOBs; WHS +0.008 ns is the only hold margin.
- The ordering code suffix (sector size digit) is not known from the schematic; step 3 reads it from the chip.
- No fault-injection set for `board_ctrl`/UART beyond the one mutation and the QUAD=0 negative test.
- `boolean_core` defaults duplicate the 15M parameters (checked by `impl_top.py`, not generated).
- BRAM margin is still 1 BRAM36 (67/68).
