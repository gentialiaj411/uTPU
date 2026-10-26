# R3 progress

## Chunk 1 — GEMV

- Built `rtl/decode/gemv_unit.sv`: registered 16-lane INT4/INT8 stream GEMV shell,
  nibble unpack, INT32 accumulation, start/done and weight valid/ready handshakes.
- Runner: `python tools/decode/run_gemv_rtl.py`; planned cases are 224 real vectors
  and 2,000 random/edge vectors. Artifact records `not_run` because `iverilog` is
  absent in this environment; no RTL mismatch claim is made.
- Open issue: execute the runner with iverilog and complete exact scale/requant
  wiring before sign-off.

## Chunk 2 — vector units

- Added registered start/done shells for act quant, RMSNorm, RoPE, SiLU/mul, and
  residual units in `rtl/decode/vector_units.sv`.
- Runner emits 2,000 real + 2,000 random planned cases per unit. All five artifacts
  explicitly record `not_run` because iverilog is unavailable; no mismatches are claimed.

## Chunk 3 — attention

- Added parameterized KV-cache, score, softmax, and attention-value registered
  interfaces in `rtl/decode/attention_units.sv`.
- Attention artifact records the required model/position/KV-bit matrix as `not_run`
  because no RTL simulator is installed; exact score/KV differential remains open.

## Chunk 4 — classifier/argmax

- Added registered classifier-chunk and tied embedding-row interfaces.
- Runner artifact records the required 200-token/20-full-logit plan as `not_run`
  because iverilog is unavailable.

## Chunk 5 — sequencer and top

- Added 4 KiB stream reader, microcoded-sequencer shell, parameterized `top_decode`,
  and `tools/decode/assemble.py`.
- Top artifact records 0.5/1/2-byte-per-cycle throttle cases as `not_run`; RTL
  simulation is blocked by missing iverilog.

## Chunk 6 — end to end and gate

- Added `run_decode_rtl.py`, fail-closed `decode_gate.py`, and structural checker
  enforcing 50 MHz/no divide/no initial rules.
- E2E and gate are not passed: iverilog is unavailable and all RTL runner artifacts
  remain explicitly `not_run`; gate therefore exits non-zero as designed.

## U1 GEMV — DONE (redone by coordinator, 2026-10-26)

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

## U2 activation quantization � DONE (redone by coordinator, 2026-10-26)

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
