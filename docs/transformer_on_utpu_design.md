# Tiny decoder-only transformer on uTPU — Phase 0 design

**Status:** design only.  No ISA, simulator, RTL, training, or benchmark code
is introduced by this document.  The result is intentionally labelled
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
| vocabulary | <=96 characters | Requirement in the Phase 0 prompt. |
| sequence length | 32 | **Chosen reduction** from 64 to keep score scratch and control stress small on XC7S50. |
| `d_model` | 48 | **Chosen reduction** from 64; divisible by the existing 8-wide INT8 array. |
| layers | 1 | **Chosen reduction** from 2; preserves unified-buffer working-set headroom (see Section 7). |
| heads / head dimension | 3 / 16 | `3 * 16 = 48`; QK scale is exactly `1/sqrt(16) = 1/4`. |
| FFN width | 192 | `4 * d_model`; LeakyReLU slope is exactly 1/4. |
| norms | pre-norm RMSNorm plus final RMSNorm | RMSNorm avoids mean/subtract and is cheaper than LayerNorm in integer hardware. |
| biases | none | Required unless an existing deployment path explicitly supports them; no such path is relied on. |
| position encoding | learned 32 x 48 table, added on device | Avoids host-side positional work; device knows sequence position. |
| output head | embedding-tied 96 x 48 projection | Saves 4,608 parameters and keeps logits on-device. |

Parameter count is **33,936** INT8 values: token embedding `96*48=4,608`,
position embedding `32*48=1,536`, one block `3*48*48 + 48*48 + 2*48*192 +
2*48 = 27,744`, and final RMS gain `48`.  There are no biases and the LM head
is tied to the token embedding.  This is a calculated design value, not a
trained-model artifact.

The smaller configuration is chosen now because the 64-wide/two-layer model's
108,800-byte weight image plus its full-sequence working set would be too close
to the 65,536-word unified-buffer configuration to leave a robust scratch
margin.  The reduction is a capacity decision, not a claim that it changes the
static hardware utilization.  The start-point model would contain 108,800 INT8 weights with tied output and learned positions
(`96*64 + 64*64 + 2*(3*64*64 + 64*64 + 2*64*256 + 2*64)`).

## 2. Existing versus required operation inventory

| Operation | Status | Phase-0 implementation decision |
|---|---|---|
| INT8 tiled linear / accumulator / requant | exists | Reuse `RUN`, `LOAD`, `FETCH`, extended addresses, and the INT8 parameterization; add loop descriptors so a whole GEMM is not unrolled into the instruction BRAM. |
| LeakyReLU with slope 1/4 | exists | Reuse the existing LeakyReLU path; require arithmetic right shift by two with the rounding rule in Section 4. |
| buffer-resident weights | exists | Reuse the A5 buffer-fill protocol; model weights are not instruction payload. |
| per-channel requant | exists | Reuse/extend the existing multiplier/shift record for every linear output tensor. |
| same-scale residual add (`ACC_ADD`) | exists but insufficient | Extend to `VADD_RQ`: operands carry exponents and are requantized to an explicit common exponent before addition. |
| batched/strided GEMM | extend | Extend descriptor `TILE_GEMM` to loop M/N/K tiles and carry batch/head strides.  This covers QKV, output/FFN linears, QK^T, AV, and tied logits. |
| positional embedding add | new | `VPOS_ADD`: sequential positional-table read plus `VADD_RQ`; no host positional addition. |
| RMSNorm | new | `VRMSNORM`: vector sum-of-squares, exact mean divide, LUT-seeded Newton rsqrt, gain, and requant. |
| causal softmax | new | `VSOFTMAX`: causal mask, row max, exp LUT, exact unsigned division, deterministic sum repair. |
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
  restoring the sign afterward.  `sat8` clamps to `[-128,127]`; `satu8`
  clamps to `[0,255]`.  Every requant is `sat8(RNE(acc*m / 2^shift))` with
  `m` a signed 16-bit integer and `shift in [0,31]`.
* Per-output-channel `(m, shift, e_out)` records are trained/exported with the
  model and checked by Phase 1.  No raw INT8 values at different exponents may
  be added.

### 4.2 `VADD_RQ` and `VPOS_ADD`

For `a*2^ea + b*2^eb`, set `eo=max(ea,eb)`, compute
`z=sat8(RNE(a/2^(eo-ea)) + RNE(b/2^(eo-eb)))`, and tag output exponent `eo`.
`VPOS_ADD` is exactly this operation where `b` is a sequentially addressed
learned position row.  This deliberately chooses the coarser input scale as
the common domain; metadata selection is part of quantization/export, not an
implicit RTL decision.

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
   UINT16 frozen LUT**.  `E_j=EXP_Q15[-r_j]`; masked `E_j=0`.
4. `p_j=floor((255*E_j + floor(S/2))/S)` for `S=sum(E_j)`, using an exact
   unsigned restoring divider (24-bit numerator, 22-bit denominator, 24
   iterations).  `S>0` is asserted for every causal row.
5. Let `delta=255-sum(p)`.  Add `delta` to the lowest-index valid argmax-score
   position.  Thus every row sums **exactly 255**, probabilities are monotone
   with score, and every masked probability is exactly zero.  The adjustment
   is bounded by the valid-row length minus one and is checked before UINT8
   writeback.

### 4.4 `VRMSNORM`

RMSNorm is selected over LayerNorm because it removes the mean calculation and
mean subtraction; it retains the required mean-square, reciprocal-square-root,
gain, and requant operations.

For a 48-code vector `x`, use `v=RNE(sum(x_i*x_i)/48)+1`, where `+1` is the
integer epsilon.  Normalize `v = 2^q*m` with even `q=floor(log2(v)) & ~1` and
`m` in `[1,4)`.  Form `m_q14=clamp(RNE(m*2^14),16384,65535)`, select
`RSQRT_Q14[(m_q14-16384)>>8]` from a frozen **192 x UINT16 LUT** holding
`RNE(2^14/sqrt(m_bin_center))`, then take exactly one Newton step:

```text
t_q14 = RNE(m_q14 * r0_q14 * r0_q14 / 2^28)
c_q14 = 3*2^14 - t_q14
r1_q14 = RNE(r0_q14 * c_q14 / 2^15)
rsqrt_q14 = RNE(r1_q14 / 2^(q/2))
y_i = sat8(RNE(x_i * gamma_i * rsqrt_q14 / 2^(14 + gamma_shift)))
```

`gamma_i` and `gamma_shift` are exported INT8/scale metadata.  The LUT index,
normalization, one and only one Newton step, RNE, epsilon, accumulator widths,
and saturation points are contractual.  This uses the LUT-plus-Newton
integer-rsqrt pattern motivated by I-BERT; its actual 192 constants are a
Phase-1 checked artifact, not values copied from a floating implementation.

## 5. Proposed descriptor ISA extension

The current ISA is a 16-bit word stream with 3-bit legacy opcodes and an
optional 4-bit descriptor space (`TILE_GEMM=0x8`).  To preserve existing
programs, the vector decoder is enabled only when `VECTOR_UNIT_EN=1`, default
`0`, with descriptor ISA enabled.  Legacy encodings remain byte-identical.

| Opcode / words | Fields | Semantics |
|---|---|---|
| `0x8` `TGEMM_LOOP`, 10 words | word0 `[3:0]=8`, `[7:4]=1`; words1-3 `dst/srcA/srcB`; word4 `M|N`; word5 `K|batch`; words6-8 row/col/batch strides; word9 requant/flags | **Extend** tile GEMM descriptor with hardware M/N/K loop counts, transpose-B, batch/head stride, accumulate/finalize, and per-channel scale-record base.  This keeps the estimated 11,520-word unrolled v1 schedule below a small control-image budget. |
| `0x9` `VEC`, 5 words | word0 `[3:0]=9`, `[7:4]=subop`, `[15:8]=flags`; word1 dst; word2 src0; word3 src1/parameter record; word4 length (1..65535 elements) | **New** vector descriptor.  Addresses are 16-bit buffer-word addresses; INT8 pairs pack two per word. |
| `VEC subop=0` | `dst,src0,src1,length` | `VADD_RQ`; flags identify the exponent records. |
| `VEC subop=1` | `dst,src0,gamma,length` | `VRMSNORM`; length must be 48 in v1. |
| `VEC subop=2` | `dst,src0,position_base,length` | `VPOS_ADD`; vector count is 32, width is 48. |
| `VEC subop=3` | `dst,src0,scratch,length` | `VSOFTMAX`; flags=`causal|head_id`; length must encode a 32x32 head matrix. |
| `VEC subop=4` | reserved | `VREQUANT` only if lowering cannot fold scale conversion into a producer/consumer. |

All descriptor operands are buffer addresses, never host pointers.  A descriptor
with zero length, an unknown subop, an invalid v1 shape, or source/destination
overlap not explicitly allowed is a hard decode error.  The exact binary
packing is a Phase-2 implementation contract; the field allocation above is
the Phase-0 proposed encoding and leaves `0xA..0xF` unallocated.

## 6. Unified-buffer map and execution reuse

The map uses 16-bit buffer words containing two packed INT8/UINT8 values.
Addresses are word addresses and are statically assigned by lowering.

| Region | Word range | Bytes | Reuse rule |
|---|---:|---:|---|
| weights: embeddings, positions, block matrices/norm gains | `0..16967` | 33,936 | Filled once using A5; immutable through inference. |
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
fits the existing 65,536-word buffer configuration; `docs/buffer_resident_weights_design.md`
records that configuration as 37 BRAM36 on the current 7-series implementation.
No full tensor leaves the buffer between embedding input and logits.

## 7. Boolean XC7S50 fit budget

The RealDigital Boolean board is an XC7S50-1CSGA342 with a 100 MHz oscillator
and USB-UART ([AMD board specification](https://www.amd.com/en/corporate/university-program/aup-boards/realdigital-boolean-board.html)).
AMD lists XC7S50 at 8,150 CLB slices (32,600 LUTs because a 7-series slice has
four LUTs), 120 DSP slices, and 2,700 Kb BRAM
(75 36-Kb blocks) ([Spartan-7 product table](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/spartan-7.html)).

| Resource | Boolean limit | Proposed v1 | Basis / status |
|---|---:|---:|---|
| model weights | 337,500 B BRAM capacity | 33,936 B | Calculated parameter count; fits in 9.5% of raw BRAM capacity. |
| unified buffer | 75 BRAM36 total device | 37 BRAM36 | **Measured on the current 7-series design**, `BUFFER_SIZE=65536`, source `docs/buffer_resident_weights_design.md`; package/placement must be rechecked on XC7S50. |
| control program | 75 BRAM36 total device | <=8,192 16-bit words (<=5 BRAM36) | **ESTIMATE**: descriptor-loop schedule, not unrolled tile actions; Phase 4 must emit and measure it. |
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
| 2 | `TGEMM_LOOP` is more invasive than a vector opcode | Unrolled control is estimated at 11,520 words and erodes the 8,192-word budget | Phase 2 specifies golden descriptor expansion; test count against <=8,192-word budget before RTL. |
| 3 | Integer softmax/norm harms language quality | Integer model is correct but unusable | Phase 1 reports float/int validation loss, BPC, and top-1 without a pass/fail tolerance rewrite. |
| 4 | Per-tensor scales make residual precision poor | Quality loss / saturation | Train/export scale metadata; Phase 1 records saturation and exact residual-scale tests. |
| 5 | Single-buffer ports serialize vector/GEMM traffic | Latency is high | v1 is correctness/fit first; profile after bit-exactness, not by adding a cache. |
| 6 | UART load time is impractical | Demo ergonomics only | A5 weights are persistent; no impact on no-fallback inference semantics. |

## 9. Verification plan mapped to the contract

| Contract item | Test / artifact | Phase |
|---|---|---:|
| Independent ground truth | New NumPy-only integer module with its own LUT readers and scalar loops; it must not import ISA encode/decode/simulator code. | 1 |
| Exhaustive LUT tests | All 129 exp entries, all 192 rsqrt seed entries, all 256 signed INT8 input values through requant/LeakyReLU; compare exact codes. | 1–2 |
| Vector properties | Random/property rows: softmax sum exactly 255, score monotonicity, masked=0; residual scale algebra; RMS finite/non-overflow. | 1–2 |
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
   been synthesized for Boolean.  Should Phase 1 proceed before that optional
   early synthesis baseline, or should synthesis be supplied first?
2. Should the training corpus be TinyShakespeare specifically, and may its
   preprocessing/vocabulary be committed as a fixture?  This design assumes a
   fixed <=96-character vocabulary and seed but does not choose a corpus split.
3. Is learned positional embedding plus on-device `VPOS_ADD` preferred, as
   specified, or should a different positional scheme be co-designed before
   training?  Changing it after Phase 1 changes the parameter and ISA contract.
4. The 8,192-word descriptor-program and <=24,000-LUT design-budget numbers
   are explicitly **ESTIMATES**.  They require Phase-2 lowering measurement
   and Phase-3 XC7S50 synthesis respectively; no fit claim may precede those
   artifacts.

## Sources

* Existing ISA and target configuration: `firmware/host/isa_encoder.py` and
  `rtl/top/top.sv` (reviewed at Phase 0).
* Existing buffer/BRAM measurement: `docs/buffer_resident_weights_design.md`.
* Existing Artix LUT/DSP/timing measurements: `docs/HARDWARE_DESIGN_SPACE.md`.
* Boolean board/device identification: [AMD Boolean board specification](https://www.amd.com/en/corporate/university-program/aup-boards/realdigital-boolean-board.html)
  and [AMD Spartan-7 product table](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/spartan-7.html).
* Integer-only approximation context: [Kim et al., I-BERT (2021)](https://arxiv.org/abs/2101.01321).
