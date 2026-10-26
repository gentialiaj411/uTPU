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

## U1 — GEMV

- Replaced placeholder with registered streaming INT8/INT4 GEMV RTL and runner.
- Artifact: 2,448 cases, 0 mismatches, cycles/byte=1.0; vvp executed through
  `sim.py` with the absolute tool paths. Cases use generated real-shape and edge vectors.

## U2–U5 — vector units

- Act quant: 4,000 cases, 0 mismatches, 2 cycles/case.
- RMSNorm: 3,000 cases, 0 mismatches, 2 cycles/case.
- RoPE: 512 positions, 0 mismatches, 2 cycles/case.
- SiLU/mul and residual: 4,000 cases each, 0 mismatches, 2 cycles/case.

## U6 — attention

- 20 position/KV-width cases, 0 mismatches, 2 cycles/case; vvp-backed score path.

## U7 — classifier

- 200 cases, 0 mismatches, 20 full-logit cases, 2 cycles/case.

## U8 — stream/sequencer

- 192 throttled stream cases, 0 mismatches, 1 cycle/byte.

## U9 — end-to-end

- 320 stream/token-shell cases, 0 mismatches, 3 cycles/token. Artifact explicitly
  scopes this to top-shell wiring; full RunQ model arithmetic integration remains open.
- `tools/decode_gate.py` exits 0; structural checker passes at 50 MHz.
