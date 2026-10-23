# R4 decode engine microarchitecture

This is a separate `top_decode` build, with the frozen Phase-4 top and bitstream
left untouched.  It is the safest integration boundary: the streaming-port model
and all RunQIntegerLlama arithmetic can be verified independently, while the same
RTL is parameterized for stories260K and stories15M.

## Budget and top-level interfaces

`top_decode` has a clock/reset, token-in/token-out streaming interface, and a
weight stream (`valid/ready`, 32-bit data, end-of-tensor marker).  R4 maps that
stream to QSPI; simulation uses a byte-addressed behavioral memory.  The following
is an implementation budget (xc7s50, 50 MHz core), not a device limit:

| resource | budget | use |
|---|---:|---|
| LUT | 11k | sequencer, unpack, requant, address/control |
| FF | 8k | pipeline/state |
| BRAM36 | <=70 of 75 | KV, activations, FIFOs, LUTs |
| DSP48 | 16 | 16-lane GEMV and score/AV datapaths |
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
| KV codes + scale records, seq=256 | streamed/BRAM if budget permits | 122,880 | 958,464 |
| score/probability rows | BRAM | 4,096 | 4,096 |
| exp/SiLU/RoPE LUTs | BRAM ROM | 8,192 | 8,192 |
| weight stream FIFO | BRAM | 4 KiB | 4 KiB |

The 15M KV cache cannot be resident at seq=256: the board has no external RAM and
flash is read-only. KV is retained on chip only for the selected context; weights
are streamed. Exported mixed INT4/INT8 weight bytes are 235,952 (260K) and
13,073,472 (15M).

## GEMV

The unit has 16 signed lanes, each with an INT8 multiplier and an INT4 nibble
unpacker.  A row is consumed group-by-group (INT4 groups are 32; INT8 groups are
64).  Codes multiply activation codes into INT32 accumulators; the group
`(multiplier, right_shift)` scale is applied with RNE and the group results are
merged at an INT32 boundary.  The activation requantizer performs one max
reduction/division per 64-element group, then reciprocal-multiply/RNE/saturate per
element.  No division is present in a lane.  Registered DSP inputs/outputs give
one 16-wide dot step per cycle at the actual 50 MHz core clock. This is the
minimum practical width that remains at least 2x faster than a 100 MB/s stream
for both INT8 and INT4 rows; it uses 16 DSPs and approximately 11k LUTs.

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
the byte accounting used by the JSON hardware budgets.  A 4 KiB, two-bank FIFO
(2 KiB fill/2 KiB drain) hides QSPI latency; DMA requests are strictly
increasing addresses and never seek within a token.

### Context/KV sweep and BRAM feasibility

The bounded 50k-token command is, for example:
`python -m firmware.host.numerics_v2.eval_context --artifact models/numerics_v2/gptq_qkv_wo_int8_g32.npz --context 128 --kv-bits 4 --tokens 50000 --output build/ctx128_kv4.json`.
Run it for L=64,96,128,192 and KV=8,4; externally terminated jobs are recorded
pending rather than replaced by short-run values.  BRAM36 uses 4,608 bytes and
includes KV, 24,576-byte activations, 8,192-byte LUTs, and a 4,096-byte FIFO.

| L | 15M KV8 PPL/top1/BRAM36 | 15M KV4 PPL/top1/BRAM36 |
|---:|---|---|
|64|pending / pending / 15|pending / pending / 11|
|96|pending / pending / 21|pending / pending / 16|
|128|pending / pending / 27|pending / pending / 20|
|192|pending / pending / 39|pending / pending / 29|

The same 260K accounting (weights included because they may fit) is
L64: 61/58, L96: 64/61, L128: 66/63, L192: 76/71 BRAM36 (KV8/KV4); thus
260K streams weights at L=192 and can resident-store all weights only through L=128.
Select the largest row whose measured PPL is <=1.08x its same-L float baseline.

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
| stories260K | 230,874 | 108.3 tok/s | 216.6 tok/s | 433.1 tok/s | <57k cycles/token |
| stories15M | 11,633,661 | 2.15 tok/s | 4.30 tok/s | 8.60 tok/s | <58k cycles/token |

The compute allowance assumes 50 MHz and the 16-lane GEMV; both rows are
stream-bound.  QSPI estimates are bandwidth-only and exclude command overhead.

## Verification and timing rules

Python differential tests map `fixed._linear_high` to GEMV, `_rms_high` to
RMSNorm, `_rope_high` to RoPE, `_softmax_high` to score/softmax, and
`RunQIntegerLlama._forward_hidden` to the complete controller sequence.  First
run unit vectors and random tensors, then compare every intermediate tensor and
logit for N complete tokens against `RunQIntegerLlama`; require bit-exact codes,
argmax, and KV state.  Structural checks require registered DSP boundaries,
at most one wide operation per pipeline stage, no combinational divider, bounded
FIFO back-pressure, and timing closure at 50 MHz.

## Risks and implementation chunks

1. Freeze stream packer and a behavioral-memory `top_decode`; test address traces.
2. Implement GEMV/unpack/requant and prove INT4/INT8 row tests.
3. Add RMSNorm, LUTs, RoPE, and residual path; differential-test intermediates.
4. Add KV, score/softmax/AV and causal-position tests through 256 tokens.
5. Add FFN, classifier chunking, argmax, and microcode loops; compare full tokens.
6. Parameterize 260K/15M images, run timing/resource checks, then QSPI board smoke.
