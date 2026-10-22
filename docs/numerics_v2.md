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
| token embedding code | signed INT8 | offline per-row Q7 (`peak/128`); ROM weight, not a runtime activation |
| K/V cache code | signed INT8 | dynamic per token and 8-element KV head |
| INT8 weight | signed INT8 | symmetric per output-channel/group, group 64 |
| INT4 weight | signed INT8 container, codes -7..7 | symmetric per output-channel/group, group 32/64/128 |
| residual/nonlinear boundary | signed INT32 | fine calibrated base unit, no INT8 clipping |
| RMS gamma | unsigned INT16 | UQ4.12; covers both models (observed maximum 9.86) |
| RoPE cosine/sine | signed INT16 | Q1.15 |
| attention score | signed INT32 | scale 1/64; not clipped before row-max subtraction |
| exponential/probability | unsigned INT16 | Q0.15 |
| dot-product accumulator | signed INT32 | exact integer-product sum |
| scale product/group merge | signed INT64 transient | RNE-narrowed before INT32 boundary |
| logits | signed INT32 | common fine logit base unit |

For a runtime activation group, let `peak=max(1,max(abs(x_base)))` and `S=24`.
Compute one reciprocal per group, `recip=RNE((127<<S)/peak)`, then encode every
element as `code=sat_INT8(RNE(x_base*recip/2^S))`. Decode with
`x_base=RNE(code*peak/127)`, where division by constant 127 is the fixed Q24
multiplier 132104 followed by the same RNE shift. Thus there is one variable divide
per group and only multiply/shift per element. The runtime scale record is `peak`;
`(recip,S)` may be retained by hardware. Token embeddings are immutable ROM weights
and retain their offline per-row Q7 encoding. Linear ratios and weight scales are exported as
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
- **RMSNorm:** square/sum INT32 boundary values in INT64, perform one RNE divide by
  the row width, add integer epsilon 297 (the encoding of `1e-5` in residual-base
  squared units), and compute the exact restoring `isqrt(mean<<30)`. Compute one
  Q24 reciprocal of that root per row; normalization of every element is then only
  multiply/RNE-shift. Multiply UQ4.12 gamma and convert to the norm base. A direct
  implementation costs one width divider, at most 47 restoring-square-root cycles,
  and one reciprocal divider per row; it never divides per element.
- **RoPE:** Q15 cosine/sine ROMs cover positions 0..511 and all pairs for head sizes
  8 and 48. Complex products use INT64 temporaries, RNE-shift 15, and retain INT32 dynamic
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
  `[-1024,0]`, address `exp_q15[-delta]`, sum exponent codes in INT32, compute one
  Q24 reciprocal for the row, and multiply/RNE-shift every numerator. This adds a
  row-max pass and a 2,050-byte ROM, and
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
| `rope_cos_q15_h8`, `rope_sin_q15_h8` | 16 x 2048 each | position 0..511 x pair 0..3 |
| `rope_cos_q15_h48`, `rope_sin_q15_h48` | 16 x 12288 each | position 0..511 x pair 0..23 |
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

Dynamic g64 activation quantization costs one max-absolute reduction and exactly one
Q24 reciprocal divide per group, plus a reduction/read pass before each matmul and
one 32-bit peak record per group. KV uses the smaller g8 reduction above;
attention-value reads add a max reduction and reciprocal per causal 16-token block.
All element-wise scaling is multiply plus RNE shift. Other incremental costs are the
wider score buffer and max-before-clamp ordering, two 2,050-byte LUTs, and no
additional gamma storage.

At context length 256, the following conservative per-layer operation budget counts
runtime quantizers, both RMSNorms, causal attention-value blocks, and softmax. MACs
include weight, score, and attention-value dot products; LUT reads include RoPE,
sigmoid, and exp. The final RMSNorm adds two scalar divides, one isqrt, and 1/5
activation reciprocal(s) for stories260K/stories15M respectively.

| Model | scalar divides | isqrt | max/sum reductions | MACs | LUT reads |
|---|---:|---:|---:|---:|---:|
| stories260K | 98 | 2 | 104 | 78,080 | 2,316 |
| stories15M | 151 | 2 | 155 | 1,142,784 | 2,880 |

## stories260K results

All teacher-forced rows use the same 50,000 validation tokens. Greedy agreement is
64 prompts x up to 128 generated tokens and is summarized by median first divergence
(no divergence is censored at step 128). Float PPL is 3.47472. The prior fixed INT8
path reached PPL 3.48733 (+0.36%), 97.124% teacher-forced top-1, KL 0.003241, and
median first divergence 30. The hardware reciprocal path reaches PPL 3.48506
(+0.30%), 97.290% top-1, KL 0.002911, and median first divergence 27, passing the
<=2% / >=97% gate.

INT4 max-absolute g32/g64/g128 PPL is 4.09786/4.25439/4.27088; top-1 is
80.274%/77.828%/77.596%. The one permitted MSE-clipping improvement gives PPL
3.96024/4.09073/4.11525 and top-1 82.660%/80.918%/80.310%. Exact KL, per-prompt
divergences, accumulator bounds, and provenance are in
`bench/results/numerics_v2_260k.json`.

## stories15M results and budget

The 15M evaluation uses the same 50,000-token validation slice and TRAIN-only
calibration corpus. The 32,000-way classifier is evaluated in integer chunks of
4,096 codes; chunk-local top-k and global argmax are integer merges. Float PPL is
2.73589. INT8 reaches PPL 2.73990, 98.074% teacher-forced top-1, KL 0.0016967,
and median first divergence 33.5. INT4 max-absolute g32/g64/g128 reaches
3.15241/3.24940/3.38375 PPL with 83.862%/82.306%/80.292% top-1; MSE clipping
improves these to 3.10542/3.22719/3.33410 PPL and 84.716%/82.748%/81.192% top-1.

The packed hardware budget is identical across quantizers except for packed weight
bytes (the table reports the INT8 upper bound). At sequence length 256 it is
25,846,336 weight bytes including scale pairs, 958,464 KV-cache bytes, and
16,502,628 streamed bytes per token. The bandwidth-only QSPI estimates are
1.515/3.030/6.060 tokens/s at 25/50/100 MB/s respectively; these estimates exclude
compute and command overhead. Full per-configuration budgets, KL, divergence lists,
and checkpoint/calibration hashes are in `bench/results/numerics_v2_15m.json`.
