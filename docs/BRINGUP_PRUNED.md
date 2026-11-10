# Default fast path: Boolean board with K=8192 vocabulary pruning

The default fast path is the pruned stories15M image (classifier and embedding keep the 8,192 most frequent tokens; the rest fall back to byte tokens).

**Measured 2026-11-10 (RTL simulation, strict gate passing; not yet on hardware):**
- Through the DDR flash reader + flash model: 13,559,450 cycles/token = **3.69 tok/s** at 50 MHz, 0 logit/token/argmax mismatches (`bench/results/vocab_prune_ddr_rtl.json`); unpruned 15M DDR is 21,744,985 cycles/token = 2.30 tok/s (1.60x).
- Ideal stream (compute floor): 7,734,699 cycles/token, bit-exact (`bench/results/vocab_prune_rtl.json`). The DDR run is flash-bound.
- Quality: integer bpb 0.564346 vs 0.562313 unpruned (+0.36%) (`bench/results/vocab_prune_study.json`).
- Implementation (`bench/results/pruned_ddr_impl.json`): WNS +1.317 ns, WHS +0.009 ns, 20,860 LUT, 67 BRAM36, 73 DSP, DRC errors 0.
- Built files: `build/board/pruned_ddr/boolean_top.bit` 1,587,092 bytes (sha256 `85e0ae04…`), `.mcs` sha256 `c9f5af8a…`; weights 0x280000..0x928840 = `build/vocabprune/vocab_8192.image.bin` (6,981,696 bytes, sha256 `c2dc9c29…`, the same image the RTL runs used), all weight bytes verified in the MCS.



Build or obtain the image and its params sidecar, then run the host client with
the matching map. The client checks `params.json` `VOCAB`, image size, and any
advertised image SHA-256 before opening the UART; `--image` automatically finds
`<image>.params.json`.

```powershell
python tools/vocabprune/pack_pruned_image.py `
  --vocab-keep tools/vocabprune/idmaps/idmap_8192.json `
  --output build/vocabprune/vocab_8192.image.bin
```

`VOCAB` and the classifier/image offsets are RTL build parameters, so the pruned image requires the dedicated DDR implementation and bitstream:

```powershell
python tools/board/build_bitstream.py --pruned      # runs impl_top.py --pruned (generated boolean_top_pruned wrapper, DDR=1), then write_bitstream
python tools/board/make_flash_image.py --pruned     # bitstream at 0x0, vocab_8192.image.bin at 0x280000, verified
```

These artifacts live under `build/board/pruned_ddr/` and are not interchangeable with the default 32,000-row DDR bitstream. Claimed throughput must come from the pruned QSPI DDR e2e artifact, not the ideal-stream smoke.

```powershell
python tools/board/host/utpu_chat.py --port COM5 `
  --image build/vocabprune/vocab_8192.image.bin `
  --vocab-keep tools/vocabprune/idmaps/idmap_8192.json `
  --prompt "Hello, world!" --gen 8 --verify
```

Board flash, FPGA programming, UART, and `--verify` hardware steps: **NOT RUN**. Bring up the unpruned SDR and DDR builds first (`docs/BRINGUP_BOOLEAN.md`), then this one.
