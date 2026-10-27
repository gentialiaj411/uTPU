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
| e2e stories15M KV4, 1 B/cycle | running (1 prompt, 11 steps, MAX_SEQ 128) |

Cycle counts are data-independent: the KV8 and KV4 runs produce different tokens but identical cycles.
Cycles/token are means over the positions run (up to 35); attention cost grows with position.
At 1 B/cycle the engine is above the 232,554-cycle stream bound because the sequencer pops at most
one byte per cycle, so GEMV compute between tiles is not hidden (double-buffered tiles would fix it).
