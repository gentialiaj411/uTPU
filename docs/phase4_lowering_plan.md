# Phase 4 lowering plan — v1.1 VMATMUL/VLAYOUT

## Decision and conventions

Phase 4 lowers the one-block `T=32,D=48,H=3,Dh=16,Dff=192,vocab=65`
decoder into one Boolean program.  This revision supersedes the earlier
capability STOP: the only new execution primitives are VEC subop 4
`VMATMUL` and VEC subop 5 `VLAYOUT`; neither changes the legacy PE/systolic
datapath or VEC subops 0--3.  Negative LeakyReLU is the legacy arithmetic
floor shift by two.

Addresses are 16-bit unified-buffer words.  A packed INT8/UINT8 element `e`
uses word `base+(e>>1)`, low byte for even `e`, high byte for odd `e`.
`RM[R,C]` is row-major. `LIN[R,C]` is the legacy B=4 input stream:
`e=((r/4)*C+c)*4+(r%4)`. `LFIN[R,C]` is a legacy B=4 finalize stream:
`e=((c/8)*R+r)*8+(c%8)` (the four unused batch lanes are zero). `WT[R,C]`
is the legacy weight stream: `e=((r/8)*C+c)*8+(r%8)`. VLAYOUT writes every
padding or stride-hole byte as zero. The `LIN<->RM` and `LFIN<->RM` maps are
round-trip identities on defined elements; `RM->WT` is a write-only layout
for a legacy weight load.

The immutable allocation is unchanged: tied LM `0..1727`, learned positions
`1728..2495`, QKV `2496..5951`, Wo `5952..7103`, W1 `7104..11711`, W2
`11712..16319`, gamma vectors `16320..16391`, and VEC records `16392..16967`.
It is 16,968 words, below `WEIGHT_REGION_WORDS=27,648`.

Mutable allocation: `X=16968..17735`, `R=17736..18503`, QKV legacy output
`18504..20807`, score/probability one head `20808..21319`, temporary legacy
output `21320..22087`, head-major AV `22088..22855`, hidden legacy output
`22856..25927`, records `25928..26255`, softmax scratch `26256..26767`, QKV
row-major `26768..29071`, residual/output row-major `29072..29839`, FFN-down
legacy output `29840..30607`, and padded LM logits `30608..31759`. Peak is
31,760 words, below `BUFFER_SIZE=32,768`.

## Exact lowering

`LGEMM(M,K,N)` below means existing unrolled `LOAD_W(2), LOAD_X(2),
RUN(accumulate,2)` for every 8x8 tile and B=4 token group, followed by the
existing eight-channel per-output `NOP_REQUANT_VECTOR(17)` and `RUN(finalize,
2)`. Each multiplier/shift pair is the export's output-channel pair: clamp
the signed accumulator to signed 25 bits, unsigned-16 multiply, arithmetic
right shift/floor, then INT8 saturate. `VMATMUL` uses its record scalar pair
with exactly that contract. `VLAYOUT` and every VEC operation are six words.

| Order / reference op | Program pattern | Operand/result regions | Words |
|---|---|---|---:|
| pos_add | `VEC VPOS_ADD(X,X,pos_rec,1536)` | host gathered X + positions -> X | 6 |
| rmsnorm1 | `VEC VRMSNORM(R,X,n1_rec,1536)` | X -> R (RM) | 6 |
| VEC→legacy #1 | `VEC VLAYOUT(QKV,R,rm_to_legacy_load_32x48,1536)` | R RM -> QKV physical legacy LOAD stream | 6 |
| Q/K/V projection | `LGEMM(32,48,144)` | LIN(R), QKV weights -> QKV LFIN | 5,778 |
| legacy→VEC #2 | `VEC VLAYOUT(QKV_R,QKV_L,lfin_to_rm_32x144,4608)` | QKV legacy finalize -> non-overlapping RM views Q/K/V | 6 |
| QK^T, heads 0..2 | `VEC VMATMUL(score,Q_h,qk_rec_h,1024,causal=1)` | Q `[32,16]`, K `[32,16]` transpose -> score `[32,32]` Q4.4 | 18 |
| causal softmax | `VEC VSOFTMAX(score,score,soft_rec,1024,causal=1,head=h)` | score -> UINT8 P in place; scratch `26256..26767` | 18 |
| P·V, heads 0..2 | `VEC VMATMUL(head_h,score,pv_rec_h,512,causal=1)` | unsigned P `[32,32]`, signed V `[32,16]` -> head `[32,16]` | 18 |
| head concat / VEC→legacy #3 | `VEC VLAYOUT(R,head_major,head_merge_to_legacy_load_32x48,1536)` | head-major VMATMUL output -> physical legacy LOAD head stream | 6 |
| output projection | `LGEMM(32,48,48)` | LIN head, Wo -> nextX LFIN | 1,926 |
| legacy→VEC #4 + residual1 | `VLAYOUT(nextX_R,nextX_L,lfin_to_rm_32x48,1536); VADD_RQ(nextX_R,X,res1_rec,1536)` | projected nextX + preserved X -> nextX RM | 12 |
| rmsnorm2 | `VEC VRMSNORM(X,nextX,n2_rec,1536)` | nextX -> X RM | 6 |
| VEC→legacy #5 | `VEC VLAYOUT(hidden,X,rm_to_legacy_load_32x48,1536)` | X RM -> hidden physical legacy LOAD stream | 6 |
| FFN up + LeakyReLU | `LGEMM(32,48,192,relu=1)` | LIN X, W1 -> hidden LFIN | 7,704 |
| FFN LFIN→legacy LOAD + FFN down | `VLAYOUT(X,hidden,lfin_to_legacy_load_32x192,6144); LGEMM(32,192,48)` | explicit physical handoff, W2 -> FFN-down LFIN | 7,116 |
| legacy→VEC #6 + residual2 | `VLAYOUT(R,ffn2_L,lfin_to_rm_32x48,1536); VADD_RQ(R,nextX_R,res2_rec,1536)` | FFN X + nextX -> X RM | 12 |
| final norm | `VEC VRMSNORM(R,X,final_rec,1536)` | X -> R RM | 6 |
| VEC→legacy #7 | `VEC VLAYOUT(logits,R,rm_to_legacy_load_32x48,1536)` | final norm -> LM physical legacy LOAD stream | 6 |
| tied LM head | `LGEMM(32,48,72)` | LIN R, tied LM weights -> logits LFIN; host decodes LFIN and ignores padded 7 channels | 2,889 |
| halt | `HALT` | — | 1 |

There are seven VEC/legacy boundary conversions. The FFN-up/down handoff is
not a VEC boundary: B=4 `LFIN[32,192]` has its live four lanes first in each
8-column finalize tile, exactly the input stream consumed by the next K tile;
the lowering selects that existing tile stride rather than copying it.

The previous unrolled estimate was 35,866 words, including 10,404 words for
legacy QK/AV. VMATMUL replaces those with 36 words and eight VLAYOUTs add
48 words, including the required LFIN-to-physical FFN handoff:
`35,866 - 10,404 + 36 + 48 = 25,546` words. This fits the Boolean
profile's `PROG_DEPTH=65,536`; the profile is intentionally raised to 64K so
the conservative legacy image and future instrumentation also fit.

## VMATMUL records and legality

For QK head `h`, `src0=Q_h`, `b_base=K_h`, `M=32,K=16,N=32`, strides 16/16,
`transpose_b=1`, signed/signed, `causal=1`, and scalar `scores` multiplier /
shift. It writes zero for j>i. For P·V, `src0=P`, `b_base=V_h`,
`M=32,K=32,N=16`, strides 32/16, `transpose_b=0`, unsigned/signed,
`causal=1`, and scalar `av` multiplier/shift. Its K loop ends at i+1, exactly
matching the masked zero P values without interpreting a code above 127 as a
negative number.

The frozen eight-word record is: kind 5; B word address; packed `(M-1,K-1,
N-1,transpose_b)`; packed A/B code strides; flags `(a_unsigned,b_unsigned,
causal)`; unsigned multiplier; shift `[4:0]`; zero. Bad dimensions, odd or
short strides, illegal signedness/transpose/causal combinations, out-of-range
regions, nonzero reserved fields, record alias, or any A/B/C alias are decode
errors before the first output write.

## VLAYOUT records and testable maps

The VLAYOUT record is kind 6, `(R-1,C-1)`, mode, packed source/destination
code strides, then four zero words. Modes 0--3 are `RM_TO_LIN`, `LFIN_TO_RM`,
`RM_TO_WT`, and `LIN_TO_RM`. Mode 0 is the historical four-lane `LIN` stream;
it is **not** the physical stream consumed by legacy `LOAD` and lowering must
not use it at a legacy boundary. Modes 6 (`RM_TO_LEGACY_LOAD`) and 7
(`LEGACY_LOAD_TO_RM`) are the physical boundary maps and are the only modes
used by transformer lowering. They use the formulas at the top of this plan,
including both byte lanes, tails, and padding. The lowering emits modes 6 and
7 above; mode 2 is retained for a legal buffer-resident activation-as-weight
image and mode 3 makes all defined conversions round-trip testable. Invalid
mode/dimensions/strides/ranges, all non-identity aliases, or a nonzero header
flag are decode errors before a write.

## Decisions made autonomously

* Matrix and layout dimensions use `dimension-1` encodings so 32 is legal in
  five bits; strides are code elements and must be even to preserve packed
  byte lanes.
* The LM head's padded 72-output legacy result remains LFIN until host readout;
  no gratuitous final VLAYOUT is emitted.
* VLAYOUT mode 4 is the fixed head-major AV merge to physical legacy LOAD and
  mode 5 is LFIN-to-physical legacy LOAD. They avoid prohibited in-place
  conversions and retain the frozen legacy PE datapath.
* The measured program is 25,546 words, leaving a 22% margin to 32,768;
  Phase 4 restores `PROG_DEPTH=32768` after its required re-synthesis.
