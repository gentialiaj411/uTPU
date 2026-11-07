# Speculative decoding study: stories15M / xc7s50

## Bottom line

Do not build speculative RTL from this result. The measured lossless acceptance
is too low: the best combined drafter emits only 1.032--1.043 tokens per
verify pass for k=1--6. Under the existing full-vocabulary cost model, k=1 is
the only plausible design point, and it is a small win only at 0.25 B/cycle.

All acceptance values below are **MEASURED_OFFLINE**: one bit-exact
`RunQIntegerLlama` greedy sequence was generated per prompt, then every
drafter and k was evaluated against that saved sequence. Speed values are
**DERIVED** from those measurements and the existing RTL cycle/stream data.
The pruned-vocabulary stream range is **ESTIMATED** from the requested
5.9--6.6M bytes/token range; it is not a hardware measurement.

## Reproducibility

- 50 TinyStories validation story openings, selected with fixed seed 0 and
  listed in `bench/results/specdec_sequences.json`.
- `stories15M`, KV4, maximum sequence length 128, stop at EOS. All 50 sampled
  sequences reached length 128; no EOS occurred.
- Greedy generation wall time: 628.06 s total, with per-prompt timings in the
  sequence artifact. The requested Python 3.11 path was inaccessible/not
  present; the available interpreter was
  `C:/Users/bhask/AppData/Local/Python/bin/python.exe`, Python 3.14.4,
  NumPy 2.4.4.
- The raw study and sequence artifacts are
  `bench/results/specdec_study.json` and
  `bench/results/specdec_sequences.json`.

## Acceptance: emitted tokens per verify pass

The table reports mean emitted tokens, including the bonus token, with a 95%
CI over the 50 prompt-level means. Exact accepted/emitted histograms for every
row are in `specdec_study.json` under `distribution_accepted_tokens` and
`distribution_emitted_tokens`.

| drafter | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| context n=1 | 1.126 [1.115,1.138] | 1.159 [1.144,1.174] | 1.167 [1.151,1.183] | 1.171 [1.153,1.188] | 1.172 [1.154,1.190] |
| context n=2 | 1.062 [1.053,1.071] | 1.080 [1.069,1.092] | 1.086 [1.073,1.099] | 1.089 [1.075,1.103] | 1.092 [1.077,1.106] |
| context n=3 | 1.031 [1.024,1.037] | 1.039 [1.030,1.048] | 1.043 [1.033,1.054] | 1.046 [1.035,1.056] | 1.047 [1.036,1.059] |
| context n=4 | 1.016 [1.012,1.021] | 1.021 [1.015,1.027] | 1.024 [1.017,1.030] | 1.024 [1.017,1.031] | 1.025 [1.018,1.033] |
| static bigram | 1.151 [1.139,1.163] | 1.173 [1.157,1.188] | 1.176 [1.161,1.192] | 1.176 [1.161,1.192] | 1.176 [1.161,1.192] |
| static trigram | 1.233 [1.217,1.250] | 1.305 [1.280,1.330] | 1.316 [1.289,1.343] | 1.319 [1.292,1.347] | 1.321 [1.293,1.350] |
| stories260K host draft | 1.017 [1.014,1.020] | 1.018 [1.015,1.022] | 1.018 [1.015,1.022] | 1.018 [1.015,1.022] | 1.019 [1.015,1.022] |
| combined: context n=4, then host | 1.033 [1.028,1.038] | 1.039 [1.032,1.046] | 1.042 [1.035,1.049] | 1.043 [1.035,1.050] | 1.044 [1.036,1.052] |

The combined emitted-token histograms (all passes) are: k=1 `{1:5620,
2:190}`; k=2 `{1:5609, 2:113, 3:55}`; k=3 `{1:5596, 2:116, 3:24,
4:25}`; k=4 `{1:5599, 2:112, 3:22, 4:14, 5:11}`; k=6 `{1:5593,
2:110, 3:22, 4:14, 5:3, 6:6, 7:2}`.

The static tables were built from `models/llama2c/TinyStories-train.slice.txt`
(1,048,576 bytes, 277,097 token IDs), using the last successor for repeated
keys within each story record. The bigram table has 4,504 entries / 36,032
packed bytes; the trigram table has 51,906 entries / 622,872 packed bytes.
These belong in host RAM or
flash, not scarce BRAM. The host draft UART model counts 1 command byte, 2k
candidate bytes, and a 2-byte response header: 5, 7, 9, 11, and 15 bytes per
round for k=1,2,3,4,6 at 115200 8N1.

## Speed model

The derived model is

`tok/s = tokens_per_pass * 50,000,000 /
max(stream_bytes / B_per_cycle, 12,338,619 * (k+1))`.

The full-vocabulary stream is 10,684,413 bytes/token. The following uses the
combined drafter's measured acceptance; entries are mean tok/s with the same
95% CI propagated from tokens/pass.

| k | 0.25 B/cycle | 0.5 B/cycle | 1.0 B/cycle |
|---:|---:|---:|---:|
| 1 | 1.209 [1.202,1.215] | 2.093 [2.082,2.104] | 2.093 [2.082,2.104] |
| 2 | 1.216 [1.208,1.224] | 1.404 [1.394,1.413] | 1.404 [1.394,1.413] |
| 3 | 1.056 [1.048,1.063] | 1.056 [1.048,1.063] | 1.056 [1.048,1.063] |
| 4 | 0.845 [0.839,0.851] | 0.845 [0.839,0.851] | 0.845 [0.839,0.851] |
| 6 | 0.605 [0.600,0.609] | 0.605 [0.600,0.609] | 0.605 [0.600,0.609] |

For the **ESTIMATED** pruned-vocabulary stream (5.9--6.6M bytes/token), the
mean tok/s range is:

| k | 0.25 B/cycle | 0.5 B/cycle | 1.0 B/cycle |
|---:|---:|---:|---:|
| 1 | 1.957--2.093 | 2.093 | 2.093 |
| 2 | 1.404 | 1.404 | 1.404 |
| 3 | 1.056 | 1.056 | 1.056 |
| 4 | 0.845 | 0.845 | 0.845 |
| 6 | 0.605 | 0.605 | 0.605 |

For reference, the corresponding greedy baseline is 1.170, 2.340, and
4.052 tok/s at 0.25, 0.5, and 1.0 B/cycle. Thus even k=1 loses at the two
faster full-vocabulary points; the 0.25-B/cycle gain is only about 3.2%.

For the **ESTIMATED** 5.9--6.6M-byte/token classifier stream, k=1 gives
1.957--2.093 tok/s at 0.25 B/cycle and 2.093 tok/s at 0.5/1.0 B/cycle. At
0.25 B/cycle the corresponding greedy range is 1.894--2.119 tok/s, so the
estimated k=1 change spans -1.2% to +3.3%. k=3 is 1.056 tok/s at all three
rates and does not gain. Compute and stream are therefore close only for k=1
at the slow pruned-vocabulary point; no design point has a robust gain.

The existing implementation result was rechecked in
`bench/results/boolean_top_impl.json`: 73 DSP48 of 120 and 67 BRAM36 of 75,
timing met. Time-multiplexed speculative estimates are k=1: 73 DSP and
approximately 71--75 BRAM36; k=3: 73 DSP and approximately 75--83 BRAM36.
These added-resource figures are **ESTIMATED**, not a speculative RTL build.

## Verification and status

The two existing golden `verify_k` tests pass. A new differential test runs
`verify_k` end-to-end over three saved prompts and compares every pass's
accepted/output sequence with the offline computation; it passes.

**NOT RUN:** speculative RTL simulation, speculative RTL synthesis/build,
board UART measurement, and pruned-vocabulary hardware measurement. No RTL,
constraints, decode numerics, or `decode_gate.py` files were changed.

## Recommendation

Do not build speculative RTL yet. If a follow-up build is forced, start with
k=1 time multiplexing only, and treat 1.207 tok/s [1.201,1.214] at 0.25
B/cycle as the derived full-vocabulary expectation for the combined drafter;
it is not a robust improvement over greedy decoding. The k=3 point is likely
over the BRAM limit and is slower in the derived model.
