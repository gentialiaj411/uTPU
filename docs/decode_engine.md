# R4 decode engine microarchitecture

This is a separate `top_decode` build, with the frozen Phase-4 top and bitstream
left untouched.  It is the safest integration boundary: the streaming-port model
and all RunQIntegerLlama arithmetic can be verified independently, while the same
RTL is parameterized for stories260K and stories15M.

## Budget and top-level interfaces

`top_decode` has a clock/reset, token-in/token-out streaming interface, and a
weight stream (`valid/ready`, 32-bit data, end-of-tensor marker).  R4 maps that
stream to QSPI; simulation uses a byte-addressed behavioral memory.  The following
is an implementation budget (xc7s50, 200 MHz target), not a device limit:

| resource | budget | use |
|---|---:|---|
| LUT | 18k | sequencer, unpack, requant, address/control |
| FF | 12k | pipeline/state |
| BRAM18 | 70 | KV, activations, FIFOs, LUTs |
| DSP48 | 80 | 64-lane GEMV and score/AV datapaths |
| URAM | 0 | not assumed on xc7s50 |

The frozen design remains reproducible because no source or constraints are added
to its top; `top_decode` is a separately selected synthesis top.

## Dataflow and storage

One token enters the embedding-row reader, then the following loop runs for every
layer: RMSNorm -> dynamic INT8 quantizer -> q/k/v GEMVs -> RoPE -> KV write ->
head scores -> row max/exp LUT/normalization -> AV -> wo GEMV -> residual ->
RMSNorm -> w1/w3 GEMVs -> SiLU LUT and product -> w2 GEMV -> residual.  Final
RMSNorm feeds the classifier GEMV in 4096-vocabulary chunks (512 chunks for
stories15M); the integer tree returns argmax and the next token.

| tensor/scratch | owner | 260K bytes | 15M bytes |
|---|---|---:|---:|
| embedding row (INT8) | BRAM row buffer | 64 | 288 |
| residual/norm/hidden ping-pong (INT32) | BRAM | 4,096 | 24,576 |
| q/k/v and attention scratch | BRAM | 1,536 | 8,640 |
| KV codes + scale records, seq=256 | BRAM | 122,880 | 958,464 |
| score/probability rows | BRAM | 4,096 | 4,096 |
| exp/SiLU/RoPE LUTs | BRAM ROM | 8,192 | 8,192 |
| weight stream FIFO | BRAM | 64 KiB | 64 KiB |

The model weights are not resident: the tied embedding/classifier table and all
layer matrices are read from the stream.  Exported mixed INT4/INT8 weight bytes
including scale pairs are 235,952 (260K) and 13,073,472 (15M); the latter fits the
14,155,776-byte flash payload after the bitstream reservation.

## GEMV

The unit has 64 signed lanes, each with an INT8 multiplier and an INT4 nibble
unpacker.  A row is consumed group-by-group (INT4 groups are 32; INT8 groups are
64).  Codes multiply activation codes into INT32 accumulators; the group
`(multiplier, right_shift)` scale is applied with RNE and the group results are
merged at an INT32 boundary.  The activation requantizer performs one max
reduction/division per 64-element group, then reciprocal-multiply/RNE/saturate per
element.  No division is present in a lane.  Registered DSP inputs/outputs give
one 64-wide dot step per cycle.  At 200 MHz this provision is over 100x the
stream-limited 15M rate, so QSPI, not GEMV, is the bottleneck at 25/50/100 MB/s.

## Flash layout and prefetch

The image starts with a 64-byte header (magic, model dimensions, format version,
and SHA-256 prefix), then the tied embedding/classifier table, followed by layers
in increasing layer index.  Within each layer the order is
`wq,wk,wv,wo,w1,w3,w2`, then gamma records.  A tensor is row-major: all packed
codes for row 0 (low nibble first for INT4, one byte per INT8 code), then rows in
order; after the code plane comes the scale plane, one packed 5-byte record per
`(row,group)`: little-endian signed INT32 multiplier followed by UINT8 shift.
Each tensor ends padded to a 16-byte boundary.  The tied table has one code plane;
embedding and classifier retain their respective scale planes.  This is exactly
the byte accounting used by the JSON hardware budgets.  A 64 KiB, two-bank FIFO
(32 KiB fill/32 KiB drain) hides QSPI latency; DMA requests are strictly
increasing addresses and never seek within a token.

## Control

Use a tiny microcoded sequencer rather than duplicating layer FSMs.  Instructions
are `LOAD_ROW(tensor,row)`, `GEMV(bits,groups,rows)`, `RMSNORM(gamma)`,
`REQUANT(group)`, `ROPE(head,pos)`, `KV_WRITE(layer,head,pos)`, `SCORE(head)`,
`SOFTMAX(length)`, `AV(head)`, `ADD_RESIDUAL`, `LUT(kind)`, `ARGMAX(chunk)`,
`LOOP(count,offset)`, `SET_PTR`, `WAIT_FIFO`, and `HALT`.  Parameters hold layer,
head, group size, and sequence length, so 260K/15M require only different images.

## Cycle model

| model | streamed bytes/token | 25 MB/s | 50 MB/s | 100 MB/s | compute allowance |
|---|---:|---:|---:|---:|---:|
| stories260K | 230,874 | 108.3 tok/s | 216.6 tok/s | 433.1 tok/s | <1.0M cycles/token |
| stories15M | 11,633,661 | 2.15 tok/s | 4.30 tok/s | 8.60 tok/s | <90M cycles/token |

The compute allowance assumes 200 MHz and the 64-lane GEMV; both rows are
stream-bound.  QSPI estimates are bandwidth-only and exclude command overhead.

## Verification and timing rules

Python differential tests map `fixed._linear_high` to GEMV, `_rms_high` to
RMSNorm, `_rope_high` to RoPE, `_softmax_high` to score/softmax, and
`RunQIntegerLlama._forward_hidden` to the complete controller sequence.  First
run unit vectors and random tensors, then compare every intermediate tensor and
logit for N complete tokens against `RunQIntegerLlama`; require bit-exact codes,
argmax, and KV state.  Structural checks require registered DSP boundaries,
at most one wide operation per pipeline stage, no combinational divider, bounded
FIFO back-pressure, and timing closure at 200 MHz.

## Risks and implementation chunks

1. Freeze stream packer and a behavioral-memory `top_decode`; test address traces.
2. Implement GEMV/unpack/requant and prove INT4/INT8 row tests.
3. Add RMSNorm, LUTs, RoPE, and residual path; differential-test intermediates.
4. Add KV, score/softmax/AV and causal-position tests through 256 tokens.
5. Add FFN, classifier chunking, argmax, and microcode loops; compare full tokens.
6. Parameterize 260K/15M images, run timing/resource checks, then QSPI board smoke.
