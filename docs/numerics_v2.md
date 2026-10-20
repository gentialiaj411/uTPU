# Numerics v2 integer contract

This document freezes the bit-exact software/RTL boundary. `runtime.py` and
`fixed.py` contain no floating-point constants, dtypes, conversions, or arithmetic.
Float is restricted to TRAIN-only calibration, weight export, LUT generation, and
metric conversion of final integer logits.

## Provenance and calibration

- Model: `stories260K.bin`, SHA-256 `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696`.
- Calibration: exactly the first 128 stories from bytes 0..1048575 of the pinned
  TinyStories TRAIN file, each truncated/padded to 64 tokens. The source and exact
  128-story corpus hashes are in `calibration_260k.json`.
- Evaluation: pinned TinyStories validation slice SHA-256
  `ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7`.
  Calibration is disjoint by named upstream split and by unequal SHA-256.
- Every activation tensor has one symmetric INT8 scale selected at the 99.9th
  absolute-value percentile of the 128-sequence calibration trace. K and V have
  separate calibrated scales; the INT8 KV cache stores those codes directly.

## Types and scale records

| Quantity | Stored type | Scale/interpretation |
|---|---:|---|
| token/residual/norm/Q/K/V/FFN activations | signed INT8 | one calibrated scale per tensor class |
| INT8 weights | signed INT8 | symmetric per output channel |
| INT4 weights | signed INT8 container, codes -7..7 | symmetric per output/group, group 32/64/128 |
| RMS gamma | signed INT16 | Q2.14 |
| RoPE cosine/sine | signed INT16 | Q1.15 |
| softmax exponential/probability | unsigned INT16 | Q0.15 |
| dot-product accumulator | signed INT32 | exact sum of integer products |
| multiplier product/group sum | signed INT64 | transient before RNE and saturation |
| logits | signed INT8 | common calibrated logit scale; float only in metric code |

No real scale is consumed by the forward pass. Export converts every linear ratio
`input_scale * weight_scale / output_scale` to `(signed multiplier, unsigned
right_shift)` with `ratio ~= multiplier / 2^right_shift`. INT4 applies this pair to
each group accumulator before summing groups. Embeddings and gamma are prequantized.

## Universal requantization and rounding

All power-of-two division uses signed round-to-nearest, ties-to-even (RNE): take the
absolute magnitude, shift, increment when remainder is greater than half or exactly
half with an odd quotient, then restore the sign. Requantization is exactly
`sat_RNE((integer * multiplier) / 2^right_shift)`. Saturation is inclusive
`[-128,127]`, except Q15/explicit wider values. Integer division in RMSNorm and
softmax uses the identical nearest-even tie rule. Residual branches share the same
calibrated scale, are added in INT32, then saturated through the same requant unit.

## Operators

- **RMSNorm:** INT32 sum of squares, RNE mean, epsilon of one code-square, and
  `isqrt(mean << 30)` produce a Q15 reciprocal-RMS normalization. Multiply by Q14
  learned gamma, RNE, requant, saturate INT8.
- **RoPE:** position × pair-index cosine/sine Q15 ROMs cover 512 positions and four
  pairs (head size 8). Complex products accumulate in INT32, RNE-shift 15, saturate.
- **GQA:** query head `h` reads KV head `floor(h / (n_heads/n_kv_heads))`.
  Keys/values are stored in separate INT8 caches with calibrated K/V scales.
- **Softmax:** score INT8 has scale 1/16. Subtract row maximum, clamp to [-128,0],
  address `exp_q15[-delta]`, zero causal-mask entries, sum INT32, and RNE-divide each
  numerator times 32767. Length is 1..512.
- **SiLU/SwiGLU:** `silu_i8` has all 256 INT8 input codes; its JSON records the
  calibration-derived input/output scales. The lookup result times W3 output uses
  INT32 and the universal multiplier/RNE/saturate rule.
- **Linear:** inputs and stored weight codes multiply into INT32. INT4 group partials
  are independently requanted then accumulated in INT64 and saturated once.

## LUT inventory

`generate_luts.py` is the sole generator. `luts/luts.json` records width, signedness,
depth, shape, domain, and values; matching one-word-per-line RTL files are:
`rope_cos_q15.mem` (16×2048), `rope_sin_q15.mem` (16×2048), `silu_i8.mem`
(8×256), and `exp_q15.mem` (16×129). Regeneration takes the calibrated SiLU scales.

## Accumulator proof

Every matmul calls `checked_matmul`, which asserts `K * max|a| * max|w| <=
2^31-1` before NumPy INT32 matmul. For stories260K, the largest INT8 dot is W2:
`172*127*127 = 2,774,188`; INT4 groups are at most `128*127*7 = 113,792`.
Attention score is `8*127*127 = 129,032`; attention-value is bounded by
`512*32767*127 = 2,130,641,408`. All fit signed INT32. Exact per-config proofs are
embedded in `bench/results/numerics_v2_260k.json` and asserted in tests.

## stories260K measured results

All rows use the same 50,000 validation tokens and 64 prompts × 128 greedy tokens.
Float PPL is 3.47472. INT8 is PPL 4.09686 with 6.0425% autoregressive token
agreement. INT4 max-absolute PPL/agreement is: g32 5.07909/4.5410%, g64
5.10485/4.4556%, g128 5.10360/3.8086%. The declared large-loss gate fired, so the
single permitted improvement was MSE-optimal clipping per group. Improved PPL is
5.01652, 5.05917, and 5.03443 for g32/g64/g128 respectively; agreements are
4.2603%, 4.1992%, and 4.4312%. The improvement helps PPL consistently but does not
consistently improve free-running token identity, so both metrics remain reported.
