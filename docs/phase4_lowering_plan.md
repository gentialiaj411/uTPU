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

Mutable allocation: `X=16968..17735`, `R=17736..18503`,
`QKV=18504..20807`, score/probability one head `20808..21831`, head result
`21832..22087`, `nextX=22088..22855`, hidden `22856..25927`, records/logits
`25928..26255`, softmax scratch `26256..26767`. Peak is 26,768 words, below
`BUFFER_SIZE=32,768`.

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
| VEC→legacy #1 | `VEC VLAYOUT(LIN(QKV),R,rm_to_lin_32x48,1536)` | R RM -> QKV staging LIN | 6 |
| Q/K/V projection | `LGEMM(32,48,144)` | LIN(R), QKV weights -> QKV LFIN | 5,778 |
| legacy→VEC #2 | `VEC VLAYOUT(QKV, QKV, lfin_to_rm_32x144,4608)` | QKV LFIN -> QKV RM views Q/K/V | 6 |
| QK^T, heads 0..2 | `VEC VMATMUL(score,Q_h,qk_rec_h,1024,causal=1)` | Q `[32,16]`, K `[32,16]` transpose -> score `[32,32]` Q4.4 | 18 |
| causal softmax | `VEC VSOFTMAX(score,score,soft_rec,1024,causal=1,head=h)` | score -> UINT8 P in place; scratch `26256..26767` | 18 |
| P·V, heads 0..2 | `VEC VMATMUL(head_h,score,pv_rec_h,512,causal=1)` | unsigned P `[32,32]`, signed V `[32,16]` -> head `[32,16]` | 18 |
| head concat / VEC→legacy #3 | `VEC VLAYOUT(LIN(head),head,rm_to_lin_32x48,1536)` | three head RM views concatenate -> LIN head | 6 |
| output projection | `LGEMM(32,48,48)` | LIN head, Wo -> nextX LFIN | 1,926 |
| legacy→VEC #4 + residual1 | `VLAYOUT(nextX,nextX,lfin_to_rm_32x48,1536); VADD_RQ(nextX,nextX,res1_rec,1536)` | projected nextX + preserved X -> nextX RM | 12 |
| rmsnorm2 | `VEC VRMSNORM(X,nextX,n2_rec,1536)` | nextX -> X RM | 6 |
| VEC→legacy #5 | `VEC VLAYOUT(LIN(hidden),X,rm_to_lin_32x48,1536)` | X RM -> hidden staging LIN | 6 |
| FFN up + LeakyReLU | `LGEMM(32,48,192,relu=1)` | LIN X, W1 -> hidden LFIN | 7,704 |
| FFN down | `LGEMM(32,192,48)` | hidden LFIN directly reused as legacy input (B=4 lanes/padding agree), W2 -> X LFIN | 7,110 |
| legacy→VEC #6 + residual2 | `VLAYOUT(X,X,lfin_to_rm_32x48,1536); VADD_RQ(X,X,res2_rec,1536)` | FFN X + nextX -> X RM | 12 |
| final norm | `VEC VRMSNORM(R,X,final_rec,1536)` | X -> R RM | 6 |
| VEC→legacy #7 | `VEC VLAYOUT(LIN(logits),R,rm_to_lin_32x48,1536)` | final norm -> LM input LIN | 6 |
| tied LM head | `LGEMM(32,48,72)` | LIN R, tied LM weights -> logits LFIN; host decodes LFIN and ignores padded 7 channels | 2,889 |
| halt | `HALT` | — | 1 |

There are seven VEC/legacy boundary conversions. The FFN-up/down handoff is
not a VEC boundary: B=4 `LFIN[32,192]` has its live four lanes first in each
8-column finalize tile, exactly the input stream consumed by the next K tile;
the lowering selects that existing tile stride rather than copying it.

The previous unrolled estimate was 35,866 words, including 10,404 words for
legacy QK/AV. VMATMUL replaces those with 36 words and the seven VLAYOUTs add
42 words: `35,866 - 10,404 + 36 + 42 = 25,540` words. This is 7,228 words
below `PROG_DEPTH=32,768`, so D4 keeps the Boolean profile at 32,768 words.

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
code strides, then four zero words. Modes are `RM_TO_LIN`, `LFIN_TO_RM`,
`RM_TO_WT`, and `LIN_TO_RM`. They use the formulas at the top of this plan,
including both byte lanes, tails, and padding. The lowering emits modes 0 and
1 above; mode 2 is retained for a legal buffer-resident activation-as-weight
image and mode 3 makes all defined conversions round-trip testable. Invalid
mode/dimensions/strides/ranges, all non-identity aliases, or a nonzero header
flag are decode errors before a write.

## Decisions made autonomously

* Matrix and layout dimensions use `dimension-1` encodings so 32 is legal in
  five bits; strides are code elements and must be even to preserve packed
  byte lanes.
* The LM head's padded 72-output legacy result remains LFIN until host readout;
  no gratuitous final VLAYOUT is emitted.
* `PROG_DEPTH` remains 32,768 because the re-estimate fits; raising it would
  spend Boolean BRAM without a demonstrated need.
