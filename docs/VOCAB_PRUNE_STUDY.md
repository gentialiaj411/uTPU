# Training-free vocabulary pruning study

Date: 2026-11-07. Every result below is labeled MEASURED, DERIVED, ESTIMATED, or NOT RUN.

## Decision

**GO for K=8192 under the requested study rule.** The rule is predicted throughput >= 1.3x the unpruned stream-bound estimate and integer KV4 bits/byte <= 1.01x the fresh unpruned integer baseline. K=8192 reaches 1.6156x and 0.56434584 bpb versus a 0.56231319 baseline (+0.3615%). After fixing the simulator LUT staging bug below, the pruned RTL smoke is bit-exact.

## Frequency maps (**MEASURED**)

The llama2.c tokenizer was streamed over `C:\Users\bhask\Documents\PROJECTS\uTPU-codesign\build\codesign\data\train_sample_256MiB.txt`, 268,435,178 bytes (SHA-256 `25943627765881018888a7f8a44174fc91450ee20c208535bd10e93ff9e55dbf`). It produced 71,013,221 token occurrences and 11,468 distinct IDs. Each map is ordered as original IDs 0, 1, 2; byte fallback IDs 3..258; then descending frequency with original-ID ascending as the tie-break. The maps are saved at `tools/vocabprune/idmaps/idmap_{2048,4096,6144,8192}.json`.

## Fixed-text quality (**MEASURED**)

The eval file is `C:\Users\bhask\Documents\PROJECTS\uTPU-codesign\build\codesign\eval_text.txt`, 261,818 bytes, 323 documents, SHA-256 `c8191cfe5a461bc825b3e3326a2921dfe978a63878b4c14547033b7217adea17`. Documents are split on `<|endoftext|>`; each document is scored in non-overlapping 128-token windows with a reset at each window. The denominator is all fixed-text UTF-8 bytes. Out-of-K tokens are expanded to their UTF-8 byte fallback rows before scoring. Candidate float uses the original float checkpoint and restricted kept-row softmax. Candidate integer uses the numerics_v2 export, KV4 `RunQIntegerLlama`, exact byte-expanded input, and a pruned classifier row slice.

| K | fallback of original eval tokens | float bpb | integer KV4 bpb | full argmax outside K | integer eval tokens |
|---:|---:|---:|---:|---:|---:|
| unpruned 32000 | 0% | 0.51606443 | 0.56231319 | n/a | 69,211 |
| 2048 | 3.6069% | 1.58397018 | 1.83696338 | 2.890625% | 80,079 |
| 4096 | 0.3581% | 0.62385383 | 0.68620207 | 0.078125% | 70,246 |
| 6144 | 0.0460% | 0.53126616 | 0.57929024 | 0% | 69,359 |
| 8192 | 0.00719% | 0.51802389 | 0.56434584 | 0% | 69,230 |

The unpruned float result reproduces the published 0.516064 baseline within the last displayed digits. The fresh unpruned integer result reproduces the required 0.562313 baseline exactly to the displayed precision. Greedy generation used the first 20 nonempty fixed-eval documents, each byte-reencoded for the candidate, and 64 generated steps: K=8192 had 0/1,280 full-argmax-outside steps and 0 sequence mismatches; K=6144 also had 0/1,280 and 0 mismatches but fails the bpb rule.

## Packed hardware estimate (**DERIVED**, schedule component **ESTIMATED**)

The default packer was not changed. Before and after SHA-256 of the default stories15M image: `d0ac5863710ddec7f9b3cc2d9b526c1070fb7ecf3b07f656373dfdd0aca95541`; default image size is 12,124,224 bytes. The candidate images are made by slicing tied embedding/classifier rows and calling the existing `tools/decode/pack_decode_image.py` `pack()` function.

| K | image bytes | stream bytes/token | compute cycles estimate | limiting cycles/token | predicted tok/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| unpruned 32000 | 12,124,224 | 10,684,413 | 12,338,619 **MEASURED** | 21,368,826 | 2.3399 | 1.00x |
| 2048 | 5,654,592 | 5,562,621 | 6,600,937 | 11,125,242 | 4.4943 | 1.9208x |
| 4096 | 6,096,960 | 5,912,829 | 6,993,257 | 11,825,658 | 4.2281 | 1.8070x |
| 6144 | 6,539,328 | 6,263,037 | 7,385,577 | 12,526,074 | 3.9917 | 1.7059x |
| 8192 | 6,981,696 | 6,613,245 | 7,777,898 | 13,226,490 | 3.7803 | 1.6156x |

Throughput is `50 MHz * 0.5 B/cycle / stream_bytes` when stream-bound, then `50 MHz / max(stream_bytes/0.5, compute_cycles)`. The unpruned measured ideal-stream decode is 12,338,619 cycles/token. Its accumulated RTL profile is divided by the 11 measured steps: classifier = 6,130,002 cycles/token and non-classifier = 6,208,617 cycles/token. The candidate compute estimate scales only the classifier schedule linearly with K; all non-classifier work stays measured/unpruned. The candidates remain stream-bound under this estimate.

Relevant RTL constraints: token input, token output, and classifier row index are 16-bit (`rtl/decode/top_decode.sv:35,44,46,198`), so VOCAB must fit in 65,535. Classifier GEMV uses 16-row tiles and `ceil(VOCAB/16)` (`rtl/decode/top_decode.sv:306-310,344`); all tested K values are multiples of 16. INT4 packing requires an even D (`tools/decode/pack_decode_image.py:38-40`); for this model `CLS_KB=D/2=144`, `CLS_G=ceil(D/32)=9`, and the generated `CLS_TILE_BYTES=2736` (`rtl/decode/top_decode.sv:24`, `tools/decode/pack_decode_image.py:71-76`). No RTL, constraint, decode-numerics, or decode-gate file was changed.

## RTL simulation

Root cause: the checkout had no `rtl/decode/luts/` ROM files, although `rtl/decode/vec_alu.sv:43-44` and `rtl/decode/attention.sv:109` unconditionally load `rope_cs`, `sigmoid_q15`, and `exp_q15` with relative `$readmemh` paths. The original smoke therefore compared a golden hidden state against invalid RTL ROM state. This was a harness/input staging bug, not a pruned row-order, scale-table, tile-count, or RTL-VOCAB limitation. `tools/vocabprune/run_decode_rtl.py` now materializes the packed ROMs from the read-only host LUTs before compilation. The 260K unpruned baseline was bit-exact (4 steps, 0 errors), the 260K K=272 reproducer was bit-exact (3 steps, 0 errors, 337,359 cycles/token), and the 15M INT4 K=272 reproducer was bit-exact (2 steps, 0 errors, 6,212,124 cycles/token).

The recommended K=8192 smoke was rerun with ideal stream, KV4, one prompt, and two generated steps: **MEASURED PASS**, 0 errors across 24,576 logits, 0 token mismatches, 0 index errors, and 0 argmax errors. It took 23,204,097 total cycles / 3 steps = **7,734,699 MEASURED mean cycles/token**, and 2,769.1 seconds wall time. Its stream term is 6,613,245 bytes/token / 0.5 B/cycle = 13,226,490 cycles/token, so corrected throughput is **3.7803 tok/s DERIVED** (`50e6 / max(13,226,490, 7,734,699)`). The runner recorded image SHA-256 `c2dc9c29a946b7a747f526eb8b9229944a65cb4437ed45338c3802829bc939b3`.

Exact command:

```powershell
$env:UTPU_SIM='icarus'
python tools/vocabprune/run_decode_rtl.py --vocab-keep tools/vocabprune/idmaps/idmap_8192.json --prompts 1 --gen 2
```

The default pack remains unchanged; its SHA-256 is still `d0ac5863710ddec7f9b3cc2d9b526c1070fb7ecf3b07f656373dfdd0aca95541`.

## Host-side remap (description only)

The board client should load `idmap_8192.json`. Prompt encoding runs the normal llama2 tokenizer, replaces every dropped token by the byte-fallback IDs for the UTF-8 bytes of that token piece, and sends compact IDs (`original_to_compact`) to the board. Board output is a compact ID; the client maps it through `kept_original_ids` and decodes the original tokenizer piece, with byte fallback reconstructed as bytes. `tools/board/host/utpu_chat.py` was intentionally not changed in this study.

## Reproduction

Python used: `C:\Users\bhask\Documents\PROJECTS\uTPU-codesign\build\codesign\venv\Scripts\python.exe` (Python 3.14.4, NumPy 2.5.4, PyTorch 2.11.0+cu128, CUDA visible). Icarus is `C:\iverilog\bin`, `UTPU_SIM=icarus`.

```powershell
python tools/vocabprune/study.py --maps
python tools/vocabprune/hardware.py
python tools/vocabprune/study.py --quality
python tools/vocabprune/study.py --generation
$env:UTPU_SIM='icarus'; python tools/vocabprune/run_decode_rtl.py --vocab-keep tools/vocabprune/idmaps/idmap_8192.json --prompts 1 --gen 2
python tools/vocabprune/finalize.py
python -m pytest tests/test_vocab_prune.py -q
```

## NOT RUN

- Board synthesis, implementation, physical-board throughput, and host-client implementation: **NOT RUN**.
- RTL smoke for K=2048, 4096, and 6144: **NOT RUN**; only the recommended K=8192 was run.
- Any retraining, fine-tuning, or quantization change: **NOT RUN**; this is training-free row pruning.
