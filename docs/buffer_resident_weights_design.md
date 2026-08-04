# Design option — Buffer-resident weights (on-chip, no DDR)

**Status:** A5 host protocol **implemented** (`MAGIC_BUF_FILL=0xA5`,
`BUF_FILL_EN=1`, smoke `tb_buf_fill_smoke.sv` PASS). Shipping
`BUFFER_SIZE=16384` (smallest swept size holding FC1’s 14144-word payload).

**Steady-state MNIST finding (2026-08-04):** amortizing all BSTORE out of a
second inference is **not bit-exact** on the current fused-MNIST lowering —
weight/activation buffer regions alias (first inference overwrites locations
later LOADs need). Cycles drop to ~1261 with `bstore=0` but logits mismatch.
Do **not** cite the 1326-cycle sketch as measured. Fix requires pinned weight
span in the allocator/lowering. See `bench/results/steady_state_attribution.json`.

**Goal:** keep weight tensors out of the instruction stream so the program
image is **control-only**, while weights live in the already-synthesized
unified buffer.

---

## 1. Why this is the real capacity answer (on-chip)

From `bench/results/program_word_composition.json`:

| Program | Total words | Embedded payload | Control |
|---|---:|---:|---:|
| Capstone FC1 256×196 | 43201 | **14144** | **29057** |
| Board-fit 256×256 | 53137 | **17408** | **35729** |

Today payload is UART-embedded (`BSTORE`/`STORE` immediates) and therefore
counts against `PROG_DEPTH`. If weights are uploaded **directly into the
unified buffer**, the instruction BRAM holds control only:

| Shape | Control-only image | Fits trustworthy `PROG_DEPTH` |
|---|---:|---|
| FC1 | 29057 | **yes @ 32768** (already validated close) |
| 256×256 | 35729 | needs ~36k: **Part A control compression** and/or fixed **65536** instr BRAM |

So:

- Buffer-resident weights unlock FC1 against the already-closed 32k instr depth.
- 256×256 control still needs either Part A (~6 words/tile instead of the
  control micro-op storm) or a valid 64k instr BRAM.
- Part A’s **capacity justification returns** under this option: once payload
  leaves the program image, compressing control is exactly what admits the
  large shapes. Part A was not wrong; it was insufficient *alone* while
  weights remained in-image.

DDR / external sourcing stays deferred to models that exceed **any** on-chip
budget (instr BRAM + unified buffer), e.g. ResNet-scale.

---

## 2. Buffer math (N=8 INT8 shipping datapath)

Sweep evidence (`prog_depth_sweep.json`, `BUFFER_SIZE` arm @ `PROG_DEPTH=8192`):

| BUFFER_SIZE | Banks | Bank depth | Status | BRAM tiles | LUT-as-Memory |
|---:|---:|---:|---|---:|---:|
| 4096 | 32 | 128 | closed | 21 | 0 |
| 16384 | 32 | 512 | closed | 21 | 0 |
| **65536** | 32 | 2048 | **closed** | **37** | **0** |

FC1 weight payload ≈ 14144 × 16-bit words ≈ **28 KiB** of buffer occupancy if
stored as the same 16-bit ISA words, or less if packed as raw INT8 lanes.
Activations for a single blocked tile are far smaller than the weight working
set for these FC shapes. A clean partition on a 65536-word buffer is easy:

```
[0 .. W_END)     weight region   (host fill; durable across tiles of one layer)
[W_END .. A_END) activation / partial region (tile scratch, overwritten)
[A_END .. BUFFER_SIZE) reserved / alignment
```

For FC1, even a pessimistic 20k-word weight region leaves >45k words for
activations/partials. Banking stays integral (`BANKS = NUM_COMPUTE_LANES /
ITEMS_IN_SLOT`, `BANK_DEPTH = BUFFER_SIZE / BANKS`).

---

## 3. Host protocol sketch (additive magic, legacy preserved)

Today (instruction upload):

```
A1 | len_lo | len_hi | program_bytes... | A2 (start)
```

Proposed additive path (default-off):

```
A5 | dest_lo | dest_hi | count_lo | count_hi | raw_payload_bytes...
```

- `A5` = `MAGIC_BUF_FILL` (new). Ignored unless a compile-time / generic
  `BUF_FILL_EN=1` (or always decoded but unused on old hosts).
- `dest` = unified-buffer word address (16-bit enough for BUFFER_SIZE≤65536).
- `count` = number of 16-bit words to write (same two-byte discipline as
  program length, or reuse `UPLOAD_LEN_MAX` rules).
- Payload is packed the same way as `BSTORE` body bytes today (lo/hi), but the
  FSM writes the **buffer port**, not `instr_bram`.

Legacy `A1` program upload unchanged. Old bitstreams without `A5` simply never
see the magic.

Optional: `A6` = `MAGIC_BUF_FILL_COMMIT` if a double-buffer / bank-flip is
wanted later; not required for v1.

---

## 4. RTL FSM cost (estimated)

New states (order-of-magnitude; not implemented):

| State | Role |
|---|---|
| `BUF_FILL_HEADER_STATE` | see `A5`, latch |
| `BUF_FILL_DEST_LO/HI` | latch destination address |
| `BUF_FILL_COUNT_LO/HI` | latch word count; reject 0 / `> BUFFER_SIZE-dest` |
| `BUF_FILL_BODY_STATE` | pack bytes → `buffer_wr_*`; advance dest/count |
| return | back to `UPLOAD_HEADER_STATE` or `WAIT_START_STATE` |

Roughly **5–6 states**, one write-pointer, one remaining-count — comparable to
the existing upload body path, but wired to `unified_buffer`’s host/compute
write port instead of `u_instr_bram`.

No ISA opcode change required for v1 if the compiler emits a control-only
program that `LOAD`s / `RUN`s from the pre-filled weight base address. A later
descriptor field (“weight base in buffer”) is the Part A / Phase-6 junction.

---

## 5. Interaction with existing `BSTORE`

| Path | Writes | Counts against `PROG_DEPTH`? | Counts against UART time? |
|---|---|---|---|
| Today `BSTORE` | buffer via instruction stream | **yes** (payload words in image) | yes (and 4.0 cyc/word on-chip) |
| Buffer-fill `A5` | buffer via dedicated upload | **no** | yes (UART only; no fetch tax) |
| Future widened BSTORE | buffer via ISA during run | yes if still in-image | on-chip cycle tax drops |

Rules:

- **Do not remove `BSTORE`.** Keep it for small immediates, biases, and demos.
- Compiler flag `embed_weights=False` switches large weight tensors to `A5`
  prefill + control-only program.
- During execution, `BSTORE` must not overwrite the weight region; allocator
  already tracks buffer slots — extend it with a pinned weight span.
- BSTORE **write-arm widen** remains the primary **cycle** lever on fused MNIST
  (79.7% `bstore`) and is **independent** of this capacity option. Even with
  buffer-resident weights, any remaining runtime `BSTORE` (activations,
  residuals) still benefits from the widen.

---

## 6. Multi-layer cycle-profile impact

Fused MNIST reference (`cycle_attribution_mnist.json`): `bstore` **79.67%**.
That profile is dominated by **embedded weight/activation upload in the
instruction stream**. Under buffer-resident weights:

- Weight traffic moves from on-chip `BSTORE` FSM time → host UART `A5` time
  (wall-clock still real; START→HALT **on-chip** attribution drops sharply).
- Remaining on-chip `bstore` share is activation/partial traffic only → Part A
  Amdahl on the residual control stream becomes meaningful again for capacity
  *and* a larger fraction of the shortened program.
- Quote on-chip vs wall-clock separately. UART @ 115200 for FC1’s 28 KiB is
  still ~7.5 s unless a faster host link is used; that is a **load bandwidth**
  problem, not an instruction-BRAM capacity problem.

---

## 7. Acceptance sketch (when someone implements)

1. `BUF_FILL_EN` default-off; legacy UART replay fixtures byte-identical.
2. FC1 control-only program ≤ trustworthy `PROG_DEPTH` with weights via `A5`.
3. Bit-exact vs NumPy / ISA sim with the same buffer image.
4. Regen `program_word_composition.json` with a `control_only_with_buf_fill`
   case; update `board_fit_audit` fit criterion accordingly.
5. No claim of DDR replacement; document buffer partition and max weight span.

---

## 8. Ordering vs other work

1. **Now:** BSTORE write-arm (cycle lever) — independent.
2. **Parallel design:** fix `PROG_DEPTH≥65536` upload-length truncation (RTL)
   so instr BRAM can hold control-only 256×256 if desired without Part A.
3. **Then:** buffer-resident weights (capacity) if FC1/256×256 must run without
   external memory.
4. **Part A sequencer:** re-justified for capacity once (2) or (3) lands;
   still parked until golden check 3 + encoding freeze for weight-base fields.
5. **DDR:** only when weight+activation working set exceeds on-chip buffer.
