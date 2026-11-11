# R4 decode engine microarchitecture

## As built (2026-10-27): RTL simulation and implementation only

Nothing in this section has run on hardware. No bitstream has been generated. Sources: `rtl/decode/top_decode.sv`,
`rtl/decode/attention.sv`, `rtl/decode/qspi_reader.sv`, `rtl/board/*.sv`. Progress log and per-milestone detail: `docs/r3_progress.md`
(M1 to M6). The original plan is kept, marked superseded, at the end of this file.

### Units

Each unit is bit-exact against the numerics_v2 runtime function it implements, checked against golden captures from the real
stories260K and stories15M runtimes plus directed rounding-tie and range cases, with mutation builds and fault injection
(`bench/results/decode_fault_injection.json`).

- `gemv_unit`: 16 signed lanes (`LANES = 16`), INT4 and INT8 weight rows, INT32 accumulation, shared pipelined scaler.
- `act_quant`: dynamic group activation quantizer, qmax 127, or qmax 7 when `qmax7` is set (the INT4 KV-cache write path).
- `rmsnorm`: sequential, one shared `seq_mul` plus a constant-divisor path (`div_const_pipe`).
- `vec_alu`: RoPE, SwiGLU, residual add, embedding scale.
- `attention`: on-chip KV cache, one merged memory for K and V. Codes are 8 bits per element for KV8 and 4 bits for KV4, with a
  32-bit scale per head. Score, softmax, and 16-position AV blocks.

### top_decode

- Fixed-function sequencer FSM over a per-layer step list (embed, rms, quant, wq/wk/wv, rope, KV write, attention, wo, residual,
  rms, quant, w1/w3, swiglu, quant, w2, residual; then final rms, classifier GEMV, argmax). There is no microcode ROM.
- Eight INT32 scratch buffers (X, XB, Q, K, V, H1, H3, T) in one scratch RAM.
- Weight stream: 8-bit `st_data` with valid/ready, plus a one-cycle `seek_req` with a 32-bit `seek_addr`. The data port is
  8 bits wide, not 32.
- Prefetch: a 4 KiB first-word-fall-through FIFO (`FIFO_DEPTH = 4096`), filled by a segment prefetcher that walks the four image
  segments in consumption order (embedding codes, embedding scales, layers, classifier). Flash reads overlap compute through this
  FIFO. The sequencer pops at most one byte per cycle, and GEMV compute between tiles is not hidden behind the stream.
- Classifier GEMV over the tied INT4 table, with argmax (lowest index wins ties).

### Board top (`rtl/board/`)

- `boolean_top`: 100 MHz oscillator (pin F14) through an MMCM to a 50 MHz core clock, reset, QSPI pads (STARTUPE2 drives CCLK,
  pad registers forced into the IOBs), UART (`UART_rxd` V12, `UART_txd` U11).
- `boolean_core`: `board_ctrl` UART front end (`T` runs one decode step and probes a logit, `S` is single-bit SPI pass-through to
  the flash, `V` is a link check), `top_decode`, and `qspi_reader`.
- `qspi_reader`: Quad Output Read (6Bh, 3-byte address) on an S25FL128S, SPI mode 0, SCLK = clk/2, one byte every 4 core cycles.
  The reader stops SCLK under backpressure and issues a fresh command on each seek.
- Flash map: the weight image starts at `FLASH_BASE` = 0x280000 (2.5 MiB, above the bitstream). The image budget is
  14,155,776 bytes. The stories15M image is 13,073,472 bytes.
- Image format: `tools/decode/pack_decode_image.py` writes raw 16-row tiles (per-group scales before codes) with no compression.
  The zlib1 codec exists only in `tools/decode/pack_stream.py`, a separate packer that the RTL path does not use.

### Results

Every number below is copied from the artifact named next to it. "Simulated" means RTL simulation (iverilog). "Implemented" means
Vivado place-and-route, not a bitstream.

| item | value | status | artifact |
|---|---|---|---|
| stories15M board top (`boolean_top`), xc7s50, post-route, 50 MHz | 20,735 LUT, 9,818 FF, 73 DSP, 67 BRAM36 of 75 physical (internal budget 68), WNS +1.565 ns, WHS +0.025 ns, 0 DRC errors, no bitstream | implemented | `bench/results/boolean_top_impl.json` |
| stories15M ideal stream, 1 B/cycle, KV4, 1 prompt x 8 generated | 11 steps, 352,000 logits and tokens, 0 mismatches, 13,287,867 cycles/token mean (max 13,341,867), 11,633,661 streamed bytes/token | simulated | `bench/results/decode_e2e_rtl_15m_kv4_p1.json` |
| stories260K through the QSPI flash model, KV8, 1 prompt x 8 | 10 steps, 0 mismatches, 930,680.6 cycles/token | simulated | `bench/results/decode_e2e_rtl_kv8_p1_qspi.json` |

Derived values:

- Ideal-stream throughput at 50 MHz: 50,000,000 / 13,287,867 = 3.76 tok/s. This is a simulated ideal-stream figure. It is not board
  throughput.
- Board throughput: about 1.07 tok/s. This is an ESTIMATE, not a measurement: 11,633,661 bytes/token x 4 core cycles/byte plus about
  350 cycles of seeks, at 50 MHz (`docs/r3_progress.md`, M4). The 15M configuration has not been simulated through the QSPI model,
  and no board run has been made.

Other synthesis records (out-of-context, `bench/results/decode_synth_ooc.json`): `top_decode_15m` 20,289 LUT, 9,223 FF, 73 DSP,
67 BRAM36, WNS +0.525 ns. `top_decode_260k` 15,492 LUT, 7,001 FF, 73 DSP, 12 BRAM36, WNS +0.829 ns.
KV-cache BRAM36 for stories15M at context 128 is 104 for KV8 and 56 for KV4 (`bench/results/decode_bram_budget.json`).

### Limits and open items

- BRAM36 margin is 1 (67 of 68 internal budget). The gate in `tools/decode_gate.py` enforces the 68 limit.
- Never run on hardware: STARTUPE2 start-up edges, the QUAD-enable state, and real pad and trace delays. The XDC timing assumptions
  are in `docs/r3_progress.md` (M6).
- The stories15M end-to-end run covers one prompt of 8 tokens (11 steps). The full prompt matrix ran on stories260K only.

## Original plan (superseded, 2026-10-27)

The plan below was written before R3 and is superseded by the as-built section above. Its budgets (11k LUT, 16 DSP, 1 microcoded
sequencer, 32-bit stream, zlib image, under 58k compute cycles per token) are plan targets, not results. Where they disagree with the
as-built section, the as-built section is correct.

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
The simulator packer records each plane as a deterministic `zlib1` block to keep
the behavioral image under the flash limit; the unpacker restores the raw code and
5-byte scale planes bit-exactly.  The hardware image generator may replace this
codec with the board decompressor without changing tensor order or offsets.

### Context/KV sweep and BRAM feasibility

The bounded 50k-token command is, for example:
`python -m firmware.host.numerics_v2.eval_context --artifact models/numerics_v2/gptq_qkv_wo_int8_g32.npz --context 128 --kv-bits 4 --tokens 50000 --output build/ctx128_kv4.json`.
Run it for L=64,96,128,192 and KV=8,4; externally terminated jobs are recorded
pending rather than replaced by short-run values.  BRAM36 uses 4,608 bytes and
includes KV, 24,576-byte activations, 8,192-byte LUTs, and a 4,096-byte FIFO.

| L | 15M KV8 PPL/top1/BRAM36 | 15M KV4 PPL/top1/BRAM36 |
|---:|---|---|
|64|pending / pending / 56|pending / pending / 32|
|96|pending / pending / 80 (no)|pending / pending / 44|
|128|pending / pending / 104 (no)|pending / pending / 56|
|192|pending / pending / 152 (no)|pending / pending / 80 (no)|

The same 260K accounting (weights included because they may fit) is
L64: 64/62, L96: 66/63, L128: 69/64, L192: 73/66 BRAM36 (KV8/KV4); thus
260K streams weights at L=192 and can resident-store all weights through L=128
(or L192 with INT4 KV).
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
