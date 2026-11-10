# Default fast path: Boolean board with K=8192 vocabulary pruning

The default fast path is the pruned stories15M image. The reported **2.30 -> 3.78 tok/s** improvement is pending regeneration of the strict K=8192 ideal/DDR RTL and bpb artifacts; it is not a current claim until `tools/decode_gate.py` passes.



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
python tools/board/impl_top.py --ddr --pruned
python tools/board/build_bitstream.py --ddr --pruned
python tools/board/make_flash_image.py --ddr --pruned
```

These artifacts live under `build/board/pruned_ddr/` and are not interchangeable with the default 32,000-row DDR bitstream. Claimed throughput must come from the pruned QSPI DDR e2e artifact, not the ideal-stream smoke.

```powershell
python tools/board/host/utpu_chat.py --port COM5 `
  --image build/vocabprune/vocab_8192.image.bin `
  --vocab-keep tools/vocabprune/idmaps/idmap_8192.json `
  --prompt "Hello, world!" --gen 8 --verify
```

Board flash, FPGA programming, UART, and `--verify` hardware steps: **NOT RUN**. Strict artifacts are pending regeneration in this checkout.
