# Phase 4 lowering plan — Chunk 1 capability decision

## Decision

**STOP: the requested one-program lowering cannot be implemented bit-exactly with the existing ISA/RTL.** This is not a compiler-only gap. The legacy datapath has three independent functional mismatches with the frozen integer reference, and the most favourable hypothetical unrolled schedule already exceeds `PROG_DEPTH`.

This document records the exact intended lowering where an existing operation covers the reference, and identifies the first places where that premise fails. It does not propose a fallback or alter the numerical contract.

## Target conventions and static allocation

All addresses below are unified-buffer *word* addresses and use Boolean `IsaConfig(address_width=15, compute_data_width=8, descriptor_isa=True, vector_unit=True)`. A legacy tile is 8x8 INT8 values = 32 words. Extended legacy `LOAD`, `RUN`, and `FETCH` consume two instruction words; an 8-lane per-channel requant NOP consumes 17 words (header plus eight multiplier/shift pairs).

The reference calls the token embedding tied, but the accelerator receives host-gathered token vectors. The immutable buffer therefore needs the *tiled* copy for the LM head, not a second row-major token table. The codes remain tied; only their physical packing is specialized for the legacy weight loader.

| Immutable asset | Words | Proposed words in Section-6 weight region |
|---|---:|---:|
| tiled tied LM matrix, padded `72x48` | 1,728 | `0..1727` |
| position table, packed row-major `32x48` | 768 | `1728..2495` |
| tiled QKV `144x48` | 3,456 | `2496..5951` |
| tiled output projection `48x48` | 1,152 | `5952..7103` |
| tiled FFN-up `192x48` | 4,608 | `7104..11711` |
| tiled FFN-down `48x192` | 4,608 | `11712..16319` |
| three RMS gamma vectors | 72 | `16320..16391` |
| VEC records / reserved immutable space | <=576 | `16392..16967` |
| total | **<=16,968** | `0..16967` |

Thus immutable weights fit `WEIGHT_REGION_WORDS=27,648`. This is capacity only; it does not cure the dynamic-layout blocker below.

The mutable regions are exactly the Section-6 map: `X=16968..17735`, `R=17736..18503`, `QKV=18504..20807`, score/probability `=20808..21831`, head output `=21832..22087`, next `X=22088..22855`, FFN hidden `=22856..25927`, and records/logits `=25928..26255`. `VSOFTMAX` requires a non-overlapping 512-word scratch region. The named metadata region is only 328 words, so it must use `26256..26767` from the Section-6 unused reserve. The corrected peak is consequently 26,768 words, still below `BUFFER_SIZE=32,768`, rather than the 26,256-word design estimate.

## Instruction patterns and word accounting

`GEMM(m,k)` means an unrolled legacy 8x8 blocked matrix product. For each output block and each batch-of-four: `LOAD_W(2), LOAD_X(2), RUN(compute, acc_clear on first k tile)(2)`, then `NOP_REQUANT_VECTOR(17)` once per output block and `RUN(finalize)(2)` for each batch. The favourable count is:

```
m * (8 * k * 6 + 17 + 8 * 2)
```

for a `T=32` tensor, assuming its inputs already have legacy batch-major tile layout and can use the Boolean four-column batch. Scalar score/AV records use the three-word scalar requant NOP instead. Counts omit FETCH: phase 4 compares buffer stages, so no output UART FETCH is needed inside the program.

| Reference operation, in order | Exact existing-instruction pattern | Section-6 operands / result | Requant and words | Status |
|---|---|---|---:|---|
| `pos_add` | `VEC VPOS_ADD(dst=X, src0=X, record=pos, length=1536)` | host X / position `1728..2495` -> `16968..17735` | VEC RNE record; **6** | Numerically covered, but output is row-major `[T,D]`, not legacy GEMM tile layout. |
| `rmsnorm1` | `VEC VRMSNORM(dst=QKV scratch or X, src0=X, gamma=n1, length=1536)` | X -> scratch | VEC norm record; **6** | Covered only in row-major vector layout. X must remain for residual1. |
| Q/K/V projection | `GEMM(m=18,k=6)` from norm to QKV, eight B=4 groups | static QKV weights `2496..5951`; output `18504..20807` | qkv 144 per-channel records, floor `acc*mult >> shift`; **5,778** favourable / **22,194** B=1 | B=4 source requires a transpose from VEC row-major. B=1 avoids that one mismatch but is already too large. |
| each head QK^T plus Q4.4 score requant | for every query row: four output blocks x two K blocks of `LOAD_W, LOAD_X, RUN`; scalar requant NOP; four finalizing RUNs | Q input; K must be legacy *weight* tiles; score row -> `20808..21831` | score record is scalar `(1,6)`; **1,801/head**, **5,403** total | `LOAD_W` can address mutable buffer, but no operation creates transposed K tiles. |
| causal softmax | `VEC VSOFTMAX(dst=score, src0=score, record=scratch, length=1024, causal, head_id)` | score/prob region; scratch `26256..26767` | fixed score/prob format; **6/head**, **18** total | Covered as vector-only operation; writes genuine UINT8 codes. |
| each head P·V | for every probability row: two output blocks x four K blocks of legacy sequence, scalar AV requant, then two finalizing RUNs | P from score region; V must be legacy *weight* tiles; head output | AV record scalar `(32897,23)`; **1,667/head**, **5,001** total | Not representable: V needs transpose and P is read by legacy loader as signed INT8. |
| head concat | no instruction for a view; would place three head outputs in `[T,48]` | head output / projected-input layout | **0 only if layout already matches** | Not covered. Legacy output is column-tile order and VEC/input consumers require incompatible layouts. |
| output projection | `GEMM(m=6,k=6)` | tiled Wo `5952..7103`; attention -> next X | 48 per-channel floor requant; **1,926** favourable / **7,398** B=1 | Requires attention layout conversion. |
| `residual1` | `VEC VADD_RQ(dst=next_X, src0=next_X, record.src1=X, length=1536)` | projected next-X plus preserved current X | VEC RNE exponents; **6** | Covered; no legacy residual RUN is used because that path rejects batched residuals. |
| `rmsnorm2` | `VEC VRMSNORM` | residual1 -> current-X scratch | VEC norm; **6** | Covered only in vector layout. |
| FFN-up + LeakyReLU(0.25) | `GEMM(m=24,k=6, relu=1)` | tiled W1 `7104..11711`; norm -> FFN hidden | 192 per-channel floor requant; **7,704** favourable / **29,592** B=1 | Per-channel quant is covered; LeakyReLU is not bit-exact (legacy is arithmetic floor, reference is RNE). |
| FFN-down | `GEMM(m=6,k=24)` | tiled W2 `11712..16319`; hidden -> current-X | 48 per-channel floor requant; **7,110** favourable / **28,134** B=1 | Needs layout conversion after the prior vector/legacy boundary. |
| `residual2` | `VEC VADD_RQ(dst=current_X, src0=current_X, record.src1=next_X, length=1536)` | FFN-down plus residual1 | VEC RNE exponents; **6** | Covered if its operands can be materialized. |
| final norm | `VEC VRMSNORM` | current-X -> final-norm storage | VEC norm; **6** | Covered only in vector layout. |
| tied LM head | `GEMM(m=9,k=6)` | tiled tied matrix `0..1727`; norm -> logits `25928..26255` | 65 per-channel floor requant; **2,889** favourable / **11,097** B=1 | Needs vector-to-legacy layout conversion. |
| termination | `HALT` | -- | **1** | Covered. |

The favourable count assumes every missing conversion and numeric mode exists, but does *not* charge instructions to execute them. It is `25,407` ordinary linear words + `5,403` QK words + `5,001` AV words + `54` VEC words + HALT = **35,866 words**, already **3,098 words over** the 32,768-word program memory. The existing-only B=1 alternative is worse: the five ordinary projections alone require 98,415 words before QK, AV, or VEC instructions.

## Evidence for the blockers

### A. Activation x activation GEMM is not available in the required form

There is no address-space restriction on a legacy weight load: the simulator's `_load_weights` reads an 8x8 flat tile from any unified-buffer address, and the RTL `LOAD_STATE` reads the same buffer port. Thus `LOAD_W(K_addr)` is legal in isolation. It does **not** transpose K: `_load_weights` consumes its 64 lanes row-major as weight rows, while batched `RUN` finalization writes output lanes in `[batch-column][output-row]` order.

`BSTORE` only writes a sequential instruction payload; `STORE` can only copy one complete 16-bit word; and extended-address `LOAD` only changes addressing. None permutes words or the two byte lanes. No VEC subop is transpose or strided copy. Consequently neither K -> K^T for QK^T, V -> V^T for P·V, nor the VEC row-major <-> legacy tile-major conversions can be materialized.

### B. UINT8 probabilities cannot enter the signed legacy MAC

`VSOFTMAX` correctly writes 0..255 bytes. The legacy simulator's `_unpack_compute_word` sign-extends every 8-bit lane before `_load_inputs`, and RTL declarations `loaded_input_matrix`, `compute_stream_in`, and `compute_weights_in` are all `logic signed [COMPUTE_DATA_WIDTH-1:0]`. Therefore `p=255` becomes `-1`, and every probability in 128..255 becomes negative. There is no unsigned LOAD/RUN flag or unsigned multiply mode.

Re-centering to `p-128` requires a separately accumulated correction `128 * sum(V)` and changes the stipulated operation sequence unless that correction is made an explicit new exact primitive. It cannot be claimed as an existing-op lowering.

### C. Requant is supported; LeakyReLU is not the reference operation

Per-output-channel requant is supported in 8-lane blocks. The encoder emits `encodeRequantParamsVector` as one header plus one multiplier/shift pair for each row; the RTL latches the vector and maps it by output row. The simulator and `requantization.py` first clamp to signed 25-bit, multiply by the unsigned 16-bit multiplier, arithmetic-right-shift (floor for negatives), then clamp to INT8. That agrees with the frozen linear record contract.

The legacy leaky path is different. `top.sv::apply_leaky_relu` is `x >>> ALPHA`, and `isa_simulator._run_finalize` uses `q >> alpha_shift`. The reference uses `_rshift_rne(hidden, 2)`. For example, `-1` maps to `-1` in the legacy path but to `0` under reference nearest-even rounding. Hence `RUN(relu=1)` cannot implement the exported FFN activation bit-exactly.

### D. Program and buffer fit

The immutable tiled weight image is 16,968 words or fewer and fits both the Section-6 allocated weight region and the 27,648-word Boolean limit. The corrected mutable peak is 26,768 words and fits the 32,768-word buffer.

Program memory does not fit. In particular, `top_board.sv` fixes `MAX_BATCH_COUNT=4`; using the tempting B=32 schedule in the old estimate is not executable by that RTL instance. Even a counterfactual B=4 schedule with free layout conversion and unsigned P·V is 35,866 words, so the current 32,768-word instruction memory is insufficient.

## Minimal paths forward (each requires an explicit new decision)

1. Add a tile/layout-conversion primitive that can transpose or pack 8x8 INT8 tiles between VEC row-major and legacy batch/weight layouts. It must be specified for both byte lanes and all tail tiles, RTL implemented, and fully bitmatched. A generic `VTRANSPOSE`/pack op is the smallest way to unblock QK, AV, and every vector/legacy boundary.
2. Add an unsigned-probability times signed-INT8 MAC mode (at least an unsigned 8-bit activation path with a sufficiently wide signed product), or a dedicated exact P·V primitive. This is a PE/datapath change, not a packing change.
3. Add RNE-by-two LeakyReLU semantics, preferably as a distinct operation so existing arithmetic-floor RUN behavior remains stable. A new VEC subop or explicit RUN mode requires ISA, simulator, RTL, and regression work.
4. Add control compression (a real RTL tile-GEMM/loop sequencer) or enlarge the program store and prove Boolean synthesis again. Raising `MAX_BATCH_COUNT` from 4 to 32 is also a hardware/resource change and does not remove the layout or UINT8 blockers.

No Phase-4 compiler, ISA-sim proof, RTL proof, or Phase-4 gate work should start until a chosen combination of these hardware/ISA changes has an approved numerical contract.

