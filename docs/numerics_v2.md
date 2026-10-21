# Numerics v2 integer contract

This document freezes the bit-exact software/RTL boundary implemented by
`runq_runtime.py` and `fixed.py`. Those forward-path modules contain no floating-point
literal, dtype, conversion, or arithmetic. Float is restricted to TRAIN-only
calibration, weight export, LUT generation, the diagnostic ablation, and conversion
of final integer logits for metrics.

## Provenance and calibration

- Model: `stories260K.bin`, SHA-256
  `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696`.
- Calibration is exactly 128 64-token sequences from the pinned TinyStories TRAIN
  split. `calibration_260k.json` records the dataset revision, byte range, source
  SHA-256, and exact calibration-corpus SHA-256.
- Evaluation is a pinned TinyStories validation slice with SHA-256
  `ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7`.
  The corpora are provably disjoint by named upstream split and unequal hashes.
- The old static path used each tensor's 99.9th-percentile calibration scale and
  clipped outliers. The fixed path divides those scales by 256 only to define fine
  integer base units; runtime max-absolute scaling supplies range, so that percentile
  no longer clips a forward activation.

## Types and scales

| Quantity | Stored/working type | Interpretation |
|---|---:|---|
| matmul activation code | signed INT8 | dynamic max-abs, groups of 64 |
| token embedding code | signed INT8 | dynamic per 64-element row |
| K/V cache code | signed INT8 | dynamic per token and 8-element KV head |
| INT8 weight | signed INT8 | symmetric per output-channel/group, group 64 |
| INT4 weight | signed INT8 container, codes -7..7 | symmetric per output-channel/group, group 32/64/128 |
| residual/nonlinear boundary | signed INT32 | fine calibrated base unit, no INT8 clipping |
| RMS gamma | unsigned INT16 | UQ3.13; covers observed maximum 4.425 |
| RoPE cosine/sine | signed INT16 | Q1.15 |
| attention score | signed INT32 | scale 1/64; not clipped before row-max subtraction |
| exponential/probability | unsigned INT16 | Q0.15 |
| dot-product accumulator | signed INT32 | exact integer-product sum |
| scale product/group merge | signed INT64 transient | RNE-narrowed before INT32 boundary |
| logits | signed INT32 | common fine logit base unit |

For an activation group, let `peak=max(1,max(abs(x_base)))`. Encoding is
`code=sat_INT8(RNE(x_base*128/peak))`; decoding is
`x_base=RNE(code*peak/128)`. Its scale record is the integer pair
`(peak_multiplier,right_shift=7)`. Linear ratios and weight scales are exported as
`(multiplier,right_shift)` pairs approximating `multiplier/2^right_shift`. No real
scale is consumed in the forward pass.

## Rounding and saturation

Every power-of-two division uses signed round-to-nearest, ties-to-even (RNE): shift
the magnitude, increment when the remainder is greater than half or is exactly half
with an odd quotient, then restore the sign. General positive-denominator division
uses the same tie rule. Requantization is
`sat(RNE(integer*multiplier/2^right_shift))`. INT8 saturation is inclusive
`[-128,127]`; wider operators state their range. This rule is used for linear
rescaling, dynamic quantization, RMSNorm, RoPE, SiLU, and softmax.

## Operators

- **Linear:** reduce and quantize each 64-element dynamic activation group before
  the dot product. Weight groups are 64 for INT8 and the selected 32/64/128 for
  INT4. Every code dot product is INT32; group scale products use transient INT64,
  RNE, then merge into an asserted INT32 boundary.
- **RMSNorm:** square/sum INT32 boundary values in INT64, RNE-divide by width, add
  integer epsilon 297 (the encoding of `1e-5` in residual-base squared units), and
  compute `isqrt(mean<<30)`. Normalize in Q15, multiply UQ3.13 gamma, then convert
  to the norm base. Changing the former signed Q2.14 gamma to UQ3.13 costs no extra
  storage and eliminates saturation of 15/64 final-norm channels.
- **RoPE:** Q15 cosine/sine ROMs cover positions 0..511 and four pairs for head size
  8. Complex products use INT64 temporaries, RNE-shift 15, and retain INT32 dynamic
  range. Query head `h` maps to KV head
  `floor(h/(n_heads/n_kv_heads))`.
- **KV cache:** K and V are max-abs quantized independently per token and KV head.
  This adds two 8-way max reductions, two scale records per token/head, and scale
  application on cache read. Codes remain INT8. Q15-by-INT8 dot partials satisfy the
  documented INT32 bound. Attention-value reads requantize into 16-token blocks;
  each Q15-by-INT8 block dot accumulates in INT32, then its scale product is RNE-
  narrowed and merged at the attention boundary.
- **Softmax:** produce a wide INT32 score at scale 1/64. Apply the causal mask and
  subtract the valid row maximum before any clamp. Clamp only the delta to
  `[-1024,0]`, address `exp_q15[-delta]`, sum exponent codes in INT32, and RNE-divide
  each numerator times 32767. This adds a row-max pass and a 2,050-byte ROM, and
  prevents the old absolute-score saturation.
- **SiLU/SwiGLU:** convert the INT32 W1 result to a 1/64-spaced index. The sigmoid
  Q15 LUT covers `[-8,8]`; values outside use asymptotes 0/32767. Multiply the
  original unclipped W1 value by sigmoid Q15, then multiply W3 and requantize.
  Hardware adds two comparisons, one 2,050-byte ROM, and a Q15 multiply.

## LUT inventory

`generate_luts.py` is the sole generator. `luts/luts.json` records width,
signedness, depth, shape, domain, and values; every entry has a matching RTL `.mem`:

| LUT | Width x depth | Domain |
|---|---:|---|
| `rope_cos_q15`, `rope_sin_q15` | 16 x 2048 each | position 0..511 x pair 0..3 |
| `silu_i8` | 8 x 256 | retained legacy static-path table |
| `sigmoid_q15` | 16 x 1025 | -8..8 in steps of 1/64 |
| `exp_q15` | 16 x 1025 | `exp(-index/64)`, index 0..1024 |

## Accumulator proofs and hardware cost

Before every integer code matmul, `checked_matmul` asserts
`K*max|a|*max|w| <= 2^31-1`. With K capped at 64 on the fixed INT8 path, a weight dot
is at most `64*127*127=1,032,256`; INT4 is at most
`128*127*7=113,792`. An attention score is `8*127*127=129,032`, and the conservative
Q15 probability-by-INT8 value bound is
`512*32767*127=2,130,641,408`. All fit signed INT32; the JSON embeds every proof.

Dynamic g64 activation quantization costs one max-absolute reduction and one integer
divider (or reciprocal-multiply implementation) per group, plus a reduction/read
pass before each matmul and one 32-bit scale record per group. KV uses the smaller
g8 reduction above; attention-value reads add a max reduction and scale per 16-token
block. Other incremental costs are the wider score buffer and max-before-clamp
ordering, two 2,050-byte LUTs, and no additional gamma storage.

## stories260K results

All teacher-forced rows use the same 50,000 validation tokens. Greedy agreement is
64 prompts x up to 128 generated tokens and is summarized by median first divergence
(no divergence is censored at step 128). Float PPL is 3.47472. Fixed INT8 reaches
PPL 3.48733 (+0.36%), 97.124% teacher-forced top-1, KL 0.003241, and median first
divergence 30, passing the <=2% / >=97% gate.

INT4 max-absolute g32/g64/g128 PPL is 4.09827/4.25427/4.27312; top-1 is
80.160%/77.790%/77.518%. The one permitted MSE-clipping improvement gives PPL
3.95920/4.09348/4.11821 and top-1 82.474%/80.696%/80.212%. Exact KL, per-prompt
divergences, accumulator bounds, and provenance are in
`bench/results/numerics_v2_260k.json`.
