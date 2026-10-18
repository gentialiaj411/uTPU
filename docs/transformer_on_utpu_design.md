# Tiny decoder-only transformer on uTPU — Phase 0 design

**Version:** 1.1 (Phase 4a amendment).

**Status:** v1.1 freezes the additive Phase-4a `VMATMUL` and `VLAYOUT` VEC
subops and the hardware-floor LeakyReLU contract. The result remains
**simulation-only until Phase 5**; board execution is not implied here.

## 1. Decision record and scope

The target is a full-sequence, causal, decoder-only character model.  The host
may load the token embedding vectors, upload the program/weights, receive
logits, and sample/argmax.  From the embedding vectors through logits, the
accelerator performs positional addition, every block operation, and the tied
output projection.  There is no KV cache in v1.

### Chosen v1 model

| Field | Value | Source / rationale |
|---|---:|---|
| vocabulary | parameterized; deployment envelope <=96, Phase-1 TinyShakespeare instance 65 | The public corpus fixture fixes the actual Phase-1 vocabulary. |
| sequence length | 32 | **Chosen reduction** from 64 to keep score scratch and control stress small on XC7S50. |
| `d_model` | 48 | **Chosen reduction** from 64; divisible by the existing 8-wide INT8 array. |
| layers | 1 | **Chosen reduction** from 2; preserves unified-buffer working-set headroom (see Section 7). |
| heads / head dimension | 3 / 16 | `3 * 16 = 48`; QK scale is exactly `1/sqrt(16) = 1/4`. |
| FFN width | 192 | `4 * d_model`; LeakyReLU slope is exactly 1/4. |
| norms | pre-norm RMSNorm plus final RMSNorm | RMSNorm avoids mean/subtract and is cheaper than LayerNorm in integer hardware. |
| biases | none | Required unless an existing deployment path explicitly supports them; no such path is relied on. |
| position encoding | learned 32 x 48 table, added on device | Avoids host-side positional work; device knows sequence position. |
| output head | embedding-tied `vocab x 48` projection | Saves a separate output matrix and keeps logits on-device. |

The `vocab<=96` allocation envelope is **33,936** INT8 values: token embedding `96*48=4,608`,
position embedding `32*48=1,536`, one block `3*48*48 + 48*48 + 2*48*192 +
2*48 = 27,744`, and final RMS gain `48`.  There are no biases and the LM head
is tied to the token embedding.  This is a calculated design value, not a
trained-model artifact.

The smaller configuration is chosen now because the 64-wide/two-layer model's
108,800-byte weight image plus its full-sequence working set would be too close
to the existing Boolean 32,768-word unified-buffer configuration to leave a robust scratch
margin.  The reduction is a capacity decision, not a claim that it changes the
static hardware utilization.  The start-point model would contain 108,800 INT8 weights with tied output and learned positions
(`96*64 + 64*64 + 2*(3*64*64 + 64*64 + 2*64*256 + 2*64)`).

## 2. Existing versus required operation inventory

| Operation | Status | Phase-0 implementation decision |
|---|---|---|
| INT8 tiled linear / accumulator / requant | exists | Reuse `RUN`, `LOAD`, `FETCH`, extended addresses, and the INT8 parameterization; emit the existing unrolled tile control in v1. |
| LeakyReLU with slope 1/4 | exists | Reuse the existing LeakyReLU path; require arithmetic right shift by two with the rounding rule in Section 4. |
| buffer-resident weights | exists | Reuse the A5 buffer-fill protocol; model weights are not instruction payload. |
| per-channel requant | exists | Reuse/extend the existing multiplier/shift record for every linear output tensor. |
| same-scale residual add (`ACC_ADD`) | exists but insufficient | Extend to `VADD_RQ`: operands carry exponents and are requantized to an explicit common exponent before addition. |
| activation matmul | new VEC subop 4 | `VMATMUL` serializes QK^T and P*V using buffer activations, explicit B transpose addressing, causal bounds, and signedness flags. It does not alter the PE array. |
| layout conversion | new VEC subop 5 | `VLAYOUT` serializes byte-preserving row-major/legacy-tile permutations at VEC/legacy handoffs. |
| positional embedding add | new | `VPOS_ADD`: sequential positional-table read plus `VADD_RQ`; no host positional addition. |
| RMSNorm | new | `VRMSNORM`: vector sum-of-squares, exact mean divide, LUT-seeded Newton rsqrt, gain, and requant. |
| causal softmax | new | `VSOFTMAX`: causal mask, row max, exp LUT, and exact unsigned division. |
| reshape/head split/merge | no runtime op | Lower as descriptor strides/views; reject layouts that cannot be expressed by the descriptor. |

Existing facts above are sourced by `firmware/host/isa_encoder.py` (opcodes
`STORE` through `DTYPE`, `TILE_GEMM=0x8`, 16-bit words, extended addresses),
`rtl/top/top.sv` (INT8 parameterization, A5, `ACC_ADD` datapath hooks), and
`docs/buffer_resident_weights_design.md` (A5 and 65,536-word buffer result).
“Extend” and “new” rows are design proposals.

## 3. Dataflow and legal subset

For one sequence `X[32,48]`, the device executes:

```text
X = token_embedding (host-gathered)                     # allowed host boundary
X = VPOS_ADD(X, learned_position[0:32])
for the one pre-norm block:
    R = X
    N = VRMSNORM(X)
    QKV = GEMM(N, Wqkv); view Q,K,V as [3,32,16]
    S = GEMM_STRIDED(Q, transpose(K)) * 1/4
    P = VSOFTMAX(S, causal=true)                        # one [32,32] row set/head
    A = GEMM_STRIDED(P, V); merge heads
    X = VADD_RQ(R, GEMM(A, Wo))
    R = X
    N = VRMSNORM(X)
    H = LeakyReLU(GEMM(N, W1))
    X = VADD_RQ(R, GEMM(H, W2))
X = VRMSNORM(X)
logits = GEMM(X, tied_token_embedding^T)
```

`backend_legality` in Phase 4 must accept only this fixed rank/layout family:
INT8 weights/activations; B=1; `T=32`, `D=48`, `H=3`, `Dh=16`, `Dff=192`; no
bias, dropout, GELU, arbitrary mask, cache, or dynamic shape.  It must report
the first unsupported op/layout rather than insert a fallback.  A causal mask
is generated by `VSOFTMAX`; it is not a host tensor.

## 4. Normative integer numerics

This section is the contract for the independent NumPy reference, ISA
simulator, and RTL.  All implementers must use the supplied integer LUT files,
not regenerate values using their language math library.

### 4.1 Common rules

* Activations and weights are signed INT8, range `[-128,127]`; softmax
  probabilities are UINT8, range `[0,255]`.  Each tensor has an explicit
  signed base-2 scale exponent `e`: real value = integer code `* 2^e`.
* Linear and reduction accumulators are signed 32-bit.  The largest v1 dot
  bound is `192*127*127 = 3,096,768`, safely below `2^31`; QK is
  `16*127*127 = 258,064`; a 48-lane sum of squares is `48*127*127 = 774,192`.
* Intermediate multiply for RMSNorm is signed 48-bit; vector add uses signed
  9-bit before INT8 saturation.  Any overflow outside a stated destination is
  a test failure, never wraparound.
* `RNE(x / 2^s)` means round-to-nearest, ties-to-even on signed magnitude,
  restoring the sign afterward. `sat8` clamps to `[-128,127]`; `satu8`
  clamps to `[0,255]`. Vector operations (`VADD_RQ`, `VPOS_ADD`,
  `VRMSNORM`, and their stated intermediate steps) use RNE. GEMM-output
  requant is deliberately different: clamp the signed accumulator to the
  DSP-facing signed-25-bit range `[-2^24, 2^24-1]`, multiply by the unsigned
  16-bit exported multiplier, arithmetic-right-shift by `shift in [0,31]`
  (floor for negative products), then `sat8`. This is the established
  hardware quantizer contract and is reimplemented independently by the
  integer reference.
* Per-output-channel `(m, shift, e_out)` records are trained/exported with the
  model and checked by Phase 1.  No raw INT8 values at different exponents may
  be added.
* LeakyReLU(0.25) is the legacy hardware rule: retain non-negative hidden
  codes and replace a negative code `h` by `h >>> 2` (arithmetic floor).
  This supersedes the earlier reference RNE-by-two spelling; it is the sole
  Phase-4a numerical change.
* `VMATMUL` uses the GEMM requantizer after each scalar dot: saturate its
  signed INT32 accumulator to signed 25-bit, multiply by its unsigned 16-bit
  scalar multiplier, arithmetic-right-shift by its `[0,31]` shift (floor),
  then `sat8`. `a_unsigned`/`b_unsigned` select zero-extension before the
  signed product; P*V uses unsigned A and signed B.

### 4.2 `VADD_RQ` and `VPOS_ADD`

For `a*2^ea + b*2^eb`, use the exporter-provided `eo`, which must satisfy
`eo >= max(ea,eb)` (normally `max(ea,eb)+1` to provide carry headroom), then compute
`z=sat8(RNE(a/2^(eo-ea)) + RNE(b/2^(eo-eb)))`, and tag output exponent `eo`.
`VPOS_ADD` is exactly this operation where `b` is a sequentially addressed
learned position row.  This deliberately chooses the coarser input scale as
the common domain; metadata selection is part of quantization/export, not an
implicit RTL decision.  Phase 1 must report a saturation count for each
residual-add site.

### 4.3 Causal integer softmax

This follows the integer-only approximation approach of Kim et al.,
*I-BERT: Integer-only BERT Quantization* (ICML 2021,
[arXiv:2101.01321](https://arxiv.org/abs/2101.01321)), but fixes the following
LUT/divider contract rather than inheriting an unspecified approximation.

1. Score GEMM uses the ordinary INT32 accumulator and an exact 1/4 arithmetic
   scale for `Dh=16`; export requants it to signed Q4.4 score codes.
2. For a row `i`, positions `j>i` are invalid.  Compute `r_j=clamp(s_j-max(s),
   -128,0)` for valid entries only.  Invalid entries never enter max or sum.
3. `EXP_Q15[k] = RNE(2^15 * exp(-k/16))`, for `k=0..128`, is a **129 x
   UINT16 frozen LUT**.  `E_j=EXP_Q15[-r_j]`; masked `E_j=0`.  The committed
   fixture is generated by `tools/gen_tiny_transformer_luts.py` with Decimal
   arithmetic and ties-to-even; its payload SHA-256 is verified by test.
4. `p_j=floor((255*E_j + floor(S/2))/S)` for `S=sum(E_j)`, using an exact
   unsigned restoring divider (24-bit numerator, 22-bit denominator, 24
   iterations).  `S>0` is asserted for every causal row.
5. Write `p_j` directly.  There is no delta-repair pass and no argmax tracker.
   For a row with `n` valid positions, tests assert `255-n < sum(p) <=
   255+ceil(n/2)`, score monotonicity, and exactly zero masked entries.

### 4.4 `VRMSNORM`

RMSNorm is selected over LayerNorm because it removes the mean calculation and
mean subtraction; it retains the required mean-square, reciprocal-square-root,
gain, and requant operations.

For a 48-code vector `x`, use `v=RNE(sum(x_i*x_i)/48)+1`, where `+1` is the
integer epsilon.  Normalize `v = 2^q*m` with even `q=floor(log2(v)) & ~1` and
`m` in `[1,4)`.  Form `m_q14=clamp(RNE(m*2^14),16384,65535)`, select
`RSQRT_Q14[(m_q14-16384)>>8]` from a frozen **192 x UINT16 LUT** holding
`RNE(2^14 / sqrt(m_center_i))`, where
`m_center_i=(16384 + 256*i + 128)/2^14`, for `i=0..191`.  This bin-center
definition, Decimal/ties-to-even generator, and payload hash are the shared
reference/ISA/RTL contract.  The table is a
`RNE(2^14/sqrt(m_bin_center))`, then take exactly one Newton step:

```text
t_q14 = RNE(m_q14 * r0_q14 * r0_q14 / 2^28)
c_q14 = 3*2^14 - t_q14
r1_q14 = RNE(r0_q14 * c_q14 / 2^15)
(mantissa_q14, exponent_shift) = (r1_q14, q/2)
y_i = sat8(RNE(x_i * gamma_i * mantissa_q14 / 2^(14 + gamma_shift + exponent_shift)))
```

`gamma_i` and `gamma_shift` are exported INT8/scale metadata.  The LUT index,
normalization, one and only one Newton step, RNE, epsilon, accumulator widths,
and saturation points are contractual.  This uses the LUT-plus-Newton
integer-rsqrt pattern motivated by I-BERT; its actual 192 constants are a
Phase-1 checked artifact, not values copied from a floating implementation.

## 5. Frozen vector descriptor ISA (Phase 2)

`VEC` is an escaped six-word descriptor. Its leading escape word is `0x05A5`
(legacy `NOP`, low three bits `101`, with bit 3 clear), followed by the five
descriptor words below. This is essential: `0x9` alone aliases legacy
`FETCH(top_half=True)`. It is legal only when both `descriptor_isa` and
`vector_unit` are set; both defaults are false. `TILE_GEMM=0x8` is unchanged:
the opcode audit proves legacy STORE encoders never set bit 3, so no emitted
legacy STORE header has low nibble `0x8`; RTL STORE also ignores bit 3.

| word | Bits | Meaning |
|---|---|---|
| 0 | `[15:0]` | fixed escape `0x05A5` |
| 1 | `[3:0]` | descriptor opcode `0x9` (`VEC`), consumed only after the escape |
| 1 | `[7:4]` | subop: `0=VADD_RQ`, `1=VRMSNORM`, `2=VPOS_ADD`, `3=VSOFTMAX`, `4=VMATMUL`, `5=VLAYOUT`; `6..15` are decode errors |
| 1 | `[15:8]` | flags: bit 8 is `causal`; bits `[10:9]` are `head_id`; bits `[15:11]` are `param_kind` |
| 2 | `[15:0]` | `dst`, a unified-buffer **word** address |
| 3 | `[15:0]` | `src0`, a unified-buffer word address |
| 4 | `[15:0]` | parameter-record word address (all subops) |
| 5 | `[15:0]` | `length`, in packed-code elements, never words |

The buffer convention is the existing `_pack_compute_word` implementation in
`firmware/host/isa_simulator.py`: for `compute_data_width=8`, code lane `2k`
is bits `[7:0]` of buffer word `base+k` and lane `2k+1` is bits `[15:8]`.
INT8 lanes are two's-complement; VSOFTMAX writes UINT8 into the identical
low/high byte positions. All descriptor addresses are word addresses and all
v1 vector lengths are even.

### 5.1 Parameter records

Each `word3` points to an eight-word record; unused words must be zero. Word
0 is the record kind in `[3:0]` and reserved-zero in `[15:4]`. Signed
exponents and shifts are eight-bit two's-complement fields; an encoded value
outside its listed range is a decode error. Every pointer is a buffer-word
address. Header `param_kind` must equal 1, 2, 3, 4, 5, or 6 respectively for
the rows below; it is not inferred from the record.

| Subop / kind | record words | Meaning |
|---|---|---|
| VADD_RQ / 1 | 0,1,2,3 | word0 kind; word1 `src1`; word2 `[7:0]=ea`, `[15:8]=eb`; word3 `[7:0]=eo`. `ea, eb, eo` are signed exponents in `[-31,31]`; `eo >= max(ea,eb)`. |
| VRMSNORM / 2 | 0,1,2 | word0 kind; word1 `gamma_base`; word2 `[7:0]=gamma_shift` in `[0,31]`, `[15:8]=output_requant_shift` and must equal zero in v1. Gamma is exactly 48 signed INT8 codes at `gamma_base`; the contractual output computation is Section 4.4. |
| VPOS_ADD / 3 | 0,1,2,3 | word0 kind; word1 `position_base` for a packed `32x48` INT8 row-major table; word2 has `ea`/`eb` in low/high bytes; word3 low byte is `eo`, with the same exponent rules as VADD_RQ. |
| VSOFTMAX / 4 | 0,1 | word0 kind; word1 `scratch_base` (reserved for the RTL implementation, must be a valid non-overlapping 1024-code region in v1). The frozen Q4.4 score and UINT8 probability formats need no exponent field. |
| VMATMUL / 5 | 0..6 | word0 kind; word1 `b_base`; word2: `M-1[4:0]`, `K-1[9:5]`, `N-1[14:10]`, `transpose_b[15]`; word3: `a_row_stride[7:0]`, `b_row_stride[15:8]`, in code elements; word4 flags bit0 `a_unsigned`, bit1 `b_unsigned`, bit2 `causal`, all other bits zero; word5 scalar unsigned multiplier; word6 shift `[4:0]`; word7 zero. Dimensions are 1..32, every row stride is even and at least its logical row width. |
| VLAYOUT / 6 | 0..3 | word0 kind; word1: `R-1[4:0]`, `C-1[12:5]` (R=1..32, C=1..256); word2 layout `mode[2:0]`, all upper bits zero; word3 source/destination row stride in code elements (`src[7:0]`, `dst[15:8]`); words4..7 zero. Strides are even and at least C. |

### 5.2 Subop shapes, aliasing, and decode errors

`VADD_RQ` has `length = 48*n`, `1 <= n <= 32`; its second input is record
word1. `VRMSNORM` has `length = 48*n`, `1 <= n <= 32`, and independently
normalizes each contiguous 48-code row. `VPOS_ADD` is exactly `length=1536`
(`32x48`). `VSOFTMAX` is exactly `length=1024`, one row-major `32x32` head;
each row uses `i+1` valid entries because `causal=1`.

Exact whole-region in-place aliases `dst==src0` are allowed for all subops;
VADD_RQ also allows `dst==src1`. Any other source/destination partial overlap
is a decode error. Position and gamma tables, parameter records, and softmax
scratch must not overlap a destination or a mutable source. The decoder must
raise (never silently execute) for: VEC without both config flags; zero
length; a truncated descriptor; an unknown/reserved subop or `param_kind`;
nonzero reserved record bits/words; invalid signed ranges or pointer extent; a
v1 shape violation; illegal flags; bad record kind; any prohibited overlap;
or a VSOFTMAX `head_id > 2`/noncausal request. VMATMUL accepts only header
`causal` (header `head_id=0`, all other header flags zero); causal is legal
for QK (`transpose_b=1`, `M=N`) or P*V (`a_unsigned=1`, `transpose_b=0`,
`M=K`), with all dimensions at most 32. It computes
`C[i,j]=requant(sum(k=0..K-1) A[i,k]*B[k,j])`, or, with transpose set,
`B[j,k]`; causal rows write zero for `j>i`, and additionally bound the K loop
to `i+1` for P*V (`a_unsigned=1`, `transpose_b=0`). VMATMUL forbids every
source/destination/B/record overlap. VLAYOUT accepts no header flags and
forbids overlap except the exact identity mode. Its modes are: 0 row-major
`[R,C]` to legacy input tile stream, 1 legacy finalize tile stream to
row-major, 2 row-major to legacy weight-tile stream, and 3 legacy input tile
stream to row-major. Modes 6 and 7 are row-major to/from the physical legacy
`LOAD` stream and require `R%4==0,C%8==0`; its index is
`floor(r/4)*C*4 + floor(c/8)*32 + (r%4)*8 + (c%8)`. For all modes element `(r,c)` occupies packed word
`base + floor(index/2)`, low byte for even index and high byte for odd index;
row-major index is `r*src_stride+c`. Legacy input index is
`(floor(r/4)*C + c)*4 + (r%4)`; legacy finalize index is
`(floor(c/8)*R + r)*8 + (c%8)`; legacy weight-tile index is
`(floor(r/8)*C + c)*8 + (r%8)`. Unused batch lanes, incomplete 8-row/8-column
tiles, and stride holes are explicitly written as zero. The inverse pairs
are exact round trips on defined, unpadded elements.

V1 GEMMs use existing unrolled tile control: the **11,520-word ESTIMATE** fits
`BoardConfig.boolean_xc7s50().prog_depth == 65,536`. `TGEMM_LOOP` remains a
future control-size optimization, not a v1 ISA extension.

## 6. Unified-buffer map and execution reuse

The map uses 16-bit buffer words containing two packed INT8/UINT8 values.
Addresses are word addresses and are statically assigned by lowering.

| Region | Word range | Bytes | Reuse rule |
|---|---:|---:|---|
| weights: embeddings, positions, block matrices/norm gains | `0..16967` | 33,936 | Allocation envelope for `vocab<=96`; the 65-symbol Phase-1 instance has 32,448 values and leaves the remainder unused. Filled once using A5; immutable through inference. |
| current `X` | `16968..17735` | 1,536 | Input / block result ping-pong. |
| residual `R` | `17736..18503` | 1,536 | Preserved until corresponding residual add. |
| Q, K, V | `18504..20807` | 4,608 | Reused after AV completes. |
| score / probability, one head | `20808..21831` | 2,048 | Heads execute serially; causal score is overwritten by probability. |
| head output | `21832..22087` | 512 | Written into the merged-attention destination. |
| next/output `X` | `22088..22855` | 1,536 | Reused as output-projection and residual destination. |
| FFN hidden | `22856..25927` | 6,144 | Reused by the second FFN GEMM. |
| logits/scratch/scale records | `25928..26255` | 656 | 96 logits plus conservative metadata/scratch reserve. |
| unused reserve | `26256..65535` | 78,560 | Reserved for alignment, implementation scratch, and debug traces. |

The maximum planned live allocation is **26,256 words / 52,512 bytes**.  It
fits `BoardConfig.boolean_xc7s50().buffer_size == 32,768` with 6,512 words of
headroom.  The Boolean profile is the authoritative v1 target; it records a
46/75 BRAM36 memory-bit extrapolation in `firmware/host/board_config.py`.
No full tensor leaves the buffer between embedding input and logits.

## 7. Boolean XC7S50 fit budget

The RealDigital Boolean board is an XC7S50-1CSGA324 with a 100 MHz oscillator
and USB-UART ([AMD board specification](https://www.amd.com/en/corporate/university-program/aup-boards/realdigital-boolean-board.html)).
AMD lists XC7S50 at 8,150 CLB slices (32,600 LUTs because a 7-series slice has
four LUTs), 120 DSP slices, and 2,700 Kb BRAM
(75 36-Kb blocks) ([Spartan-7 product table](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/spartan-7.html)).

| Resource | Boolean limit | Proposed v1 | Basis / status |
|---|---:|---:|---|
| model weights | 337,500 B BRAM capacity | 33,936 B | Calculated parameter count; fits in 9.5% of raw BRAM capacity. |
| Boolean profile | 32,768 buffer words / 65,536 program words | 26,256 live buffer words / 11,520 control words | Calculated v1 map/control estimate; source profile `BoardConfig.boolean_xc7s50`. |
| BRAM36 total | 75 | 46 | **ESTIMATE** from the existing Boolean profile’s memory-bit extrapolation in `buffer_resident_deployment._boolean_bram_estimate`; XC7S50 synthesis is still required. |
| control program | 32,768 words | 11,520 words | **ESTIMATE**: existing unrolled tile control, no loop descriptor.  Phase 4 must emit and measure it. |
| exp + rsqrt LUTs | 75 BRAM36 total device | <=1 BRAM36 | **ESTIMATE**: 129x16 + 192x16 = 642 B; allocated as one conservative RAMB36-equivalent. |
| total BRAM36 | 75 | <=43 | 37 measured buffer + 5 estimated program + 1 estimated LUT; leaves >=32 blocks but requires XC7S50 synthesis. |
| DSP48E1 | 120 | <=96 | **ESTIMATE**: existing 8x8 INT8 close uses 72 DSP on Artix; reserve <=24 for a serialized vector path.  Must be measured in Phase 3 XC7S50 synthesis. |
| CLB LUTs | 32,600 | <=24,000 design budget | **ESTIMATE / hard gate**, not a utilization result.  The existing 8x8 full top is 16,594 LUTs at MB=4 on Artix (`docs/HARDWARE_DESIGN_SPACE.md`); reserve 7,406 LUTs for the vector unit/control margin.  Phase 3 must synthesize XC7S50 and report actual LUTs <=32,600 before this row becomes “fits.” |

The existing 8x8 INT8 point uses 72 DSP and 16,594 LUTs on Artix.  The Boolean
part has 120 DSPs and 32,600 LUTs, so that point is a plausible starting
configuration, but no XC7S50 synthesis result exists.  Vector-unit area and
BRAM placement remain the primary Phase-0 feasibility risks.

## 8. Risks, ranked

| Rank | Risk | Consequence | Mitigation / gate |
|---:|---|---|---|
| 1 | Vector unit or BRAM placement exceeds Boolean resources | Cannot claim a Boolean fit | Phase 3 XC7S50 synthesis; retain no-fit status until report. |
| 2 | Integer softmax/norm harms language quality | Integer model is correct but unusable | Phase 1 reports float/int validation loss, BPC, and top-1 without a pass/fail tolerance rewrite. |
| 3 | Per-tensor scales make residual precision poor | Quality loss / saturation | Train/export per-add output exponents; Phase 1 records saturation counts and exact residual-scale tests. |
| 4 | Single-buffer ports serialize vector/GEMM traffic | Latency is high | v1 is correctness/fit first; profile after bit-exactness, not by adding a cache. |
| 5 | UART load time is impractical | Demo ergonomics only | A5 weights are persistent; no impact on no-fallback inference semantics. |

## 9. Verification plan mapped to the contract

| Contract item | Test / artifact | Phase |
|---|---|---:|
| Independent ground truth | New NumPy-only integer module with its own LUT readers and scalar loops; it must not import ISA encode/decode/simulator code. | 1 |
| Exhaustive LUT tests | All 129 exp entries, all 192 rsqrt seed entries, all 256 signed INT8 input values through requant/LeakyReLU; compare exact codes. | 1–2 |
| Vector properties | Random/property rows: `255-n < sum <= 255+ceil(n/2)`, score monotonicity, masked=0; residual scale algebra and saturation counts; RMS finite/non-overflow. | 1–2 |
| ISA bit-exactness | Reference versus decoder/simulator for every op, every block intermediate, and a full model sequence. | 2, 4 |
| RTL bit-exactness | Per-op iverilog benches versus ISA vectors, then >=8 full sequences with non-saturated outputs included. | 3, 5 |
| No fallback | One lowering test asserts `fallback_ops == []` from device embedding input through logits; negative tests name first unsupported op. | 4 |
| Accuracy honesty | `bench/results/tiny_transformer_numerics.json`: float/int validation loss, bits/char, full-validation next-char top-1, quantization metadata, saturation counters. | 1 |
| End-to-end evidence | `bench/results/tiny_transformer_utpu.json` and `_rtl.json`: model hash/config, reference/sim equality, RTL replay equality, eight input hashes, generation equality. | 5 |
| Board evidence | Later runner records exact command, bitstream hash, UART transcript, and only upgrades status after byte equality. | 5 / user-run |

Tests are written reference-first: add the independent reference expectation and
observe the absent/failed implementation test before adding an encoder,
simulator, or RTL implementation.  No tolerance applies to integer equality.

## 10. Open questions requiring review

1. **Boolean fit is unproven.** The existing 8x8 Artix result is below the
   raw XC7S50 DSP/LUT limits, but vector-unit area and BRAM placement have not
   been synthesized for Boolean.  Phase 1 proceeds without synthesis by
   approval; no fit claim may precede a measurement.
2. **Resolved:** Phase 1 uses TinyShakespeare, a fixed seed, a committed
   vocabulary/split-index fixture, and a downloader with SHA-256 validation;
   the raw corpus is not committed.
3. **Resolved:** learned positional embeddings plus on-device `VPOS_ADD` are
   the v1 positional scheme.
4. The 11,520-word control estimate, 46-BRAM36 estimate, and <=24,000-LUT
   design-budget number are explicitly **ESTIMATES**.  They require Phase-2
   lowering measurement and Phase-3 XC7S50 synthesis respectively.
5. The model/reference/export must be parameterized by `(T, D, H, Dff, L,
   vocab)`; only future backend legality pins v1 values.

## Sources

* Existing ISA and target configuration: `firmware/host/isa_encoder.py` and
  `rtl/top/top.sv` (reviewed at Phase 0).
* Existing buffer/BRAM measurement: `docs/buffer_resident_weights_design.md`.
* Existing Artix LUT/DSP/timing measurements: `docs/HARDWARE_DESIGN_SPACE.md`.
* Boolean board/device identification: [AMD Boolean board specification](https://www.amd.com/en/corporate/university-program/aup-boards/realdigital-boolean-board.html)
  and [AMD Spartan-7 product table](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/spartan-7.html).
* Integer-only approximation context: [Kim et al., I-BERT (2021)](https://arxiv.org/abs/2101.01321).
