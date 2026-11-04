# Speculative decoding study: stories15M / xc7s50

## Bottom line

The existing measured design is bandwidth-bound at the requested DDR point:
21,744,972 cycles/token at 0.5 B/cycle, versus 12,338,619 cycles/token with
the ideal stream.  The 15M placed baseline uses 73/120 DSP48 and 67/75 BRAM36.
Speculative decoding is therefore only attractive if acceptance is high enough
to amortize extra activation/KV storage.  This run did not complete the required
50-prompt acceptance sweep; no acceptance rate is fabricated here, so a build
recommendation requiring that evidence is deferred.

## Evidence and reproducibility

Measured source artifacts: `bench/results/decode_e2e_rtl_15m_kv4_p1.json` and
`bench/results/boolean_top_impl.json`.  The requested Python 3.11 path was not
present.  The available interpreter was `C:/Users/bhask/AppData/Local/Python/bin/python.exe`
(Python 3.14.4, NumPy 2.4.4).

Attempted command: `python tools/specdec/measure.py`.
Status: **NOT RUN TO COMPLETION** after shared-host contention; its intended
inputs were 50 distinct TinyStories validation openings, two generated tokens,
KV4, and k=1..4.  The full RTL speculative simulation was **NOT RUN** because
no speculative RTL was implemented.

## Cost and cycle model

The model uses `max(stream_bytes / rate, compute_cycles * (k+1))` and 50 MHz.
The measured stream size is 10,684,413 bytes/token.  Thus stream-only costs are
42,737,652, 21,368,826, and 10,684,413 cycles at 0.25, 0.5, and 1.0 B/cycle.
The existing 12,338,619-cycle compute path is multiplied by k+1 for a
time-multiplexed verifier.  These multiplied costs are **DERIVED**, not RTL
measurements.

| design point | DSP | BRAM36 | LUT | verify compute cycles | assessment |
|---|---:|---:|---:|---:|---|
| k=1, time-multiplexed | 73 est. | 71–75 est. | 22–26k est. | 24,677,238 | possible, but BRAM margin is nearly zero |
| k=3, time-multiplexed | 73 est. | 75–83 est. | 25–32k est. | 49,354,476 | likely does not fit BRAM |

Full parallel GEMV for k+1 would require roughly 73×(k+1) DSP, so k=1
already exceeds the 120-DSP budget.  Time multiplexing preserves DSP count but
exposes compute time.  The cited RTL changes are: `top_decode.sv` schedule and
scratch roles (one activation lane per candidate), `gemv_unit.sv` candidate
accumulators/input selection, `attention.sv` causal attention over draft
positions and temporary KV writes, and `board_ctrl.sv` framing.  KV writes must
be tagged by candidate position and only accepted entries committed; rejected
entries must be ignored or overwritten.  Classifier/argmax must emit one result
per candidate.  The protocol needs a draft-count plus k 16-bit draft tokens and
must return accepted-count plus accepted/fallback tokens.

## Drafters

The intended measured sweep covers context n-gram n=1..3, static train-text
bigram/trigram tables, and a host-side stories260K draft re-tokenized to the
15M tokenizer.  Static tables are logically packed as `(n token IDs + one
 successor)` using 32-bit IDs; they belong in host RAM or flash, not scarce
BRAM.  UART framing is 16-bit token IDs; at 115200 8N1 a round carrying k
drafts is approximately `1 + 2k + 2` bytes and 10 bit-times per byte.  These
are protocol-model numbers, not board measurements.  Early-exit/self-
speculation was not run and is not recommended without an independently
verified prefix-layer classifier.

## Golden model and verification

`firmware/host/numerics_v2/specdec_verify.py` adds `verify_k` without changing
the integer runtime.  It snapshots all KV arrays, verifies candidates using
`forward_argmax`, restores on the first mismatch, and replays only the accepted
prefix.  `tests/test_specdec.py` passed 2 tests, including a rejection rollback
comparison against sequential KV state and multiple k values.  Remaining work
before RTL build: add a sequential-vs-batched differential oracle, mutate each
candidate lane/causal-mask/KV-commit path, test rejection at every draft index,
and run board-protocol end-to-end tests.

## Recommendation

Do not build speculative RTL yet.  First complete the acceptance sweep on an
idle host.  If it shows a sustained high acceptance rate, start with k=1
time-multiplexed because it is the only point plausibly within the one-BRAM
margin; k=3 is currently estimated over the BRAM limit.  A real recommendation
and tok/s error bars require the missing measured acceptance distribution.
