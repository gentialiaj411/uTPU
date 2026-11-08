# Boolean board bring-up: pruned vocabulary (see BRINGUP_BOOLEAN.md for the base runbook)



Build or obtain the image and its params sidecar, then run the host client with
the matching map. The client checks `params.json` `VOCAB`, image size, and any
advertised image SHA-256 before opening the UART; `--image` automatically finds
`<image>.params.json`.

```powershell
python tools/vocabprune/pack_pruned_image.py `
  --vocab-keep tools/vocabprune/idmaps/idmap_8192.json `
  --output build/vocabprune/vocab_8192.image.bin
python tools/board/host/utpu_chat.py --port COM5 `
  --image build/vocabprune/vocab_8192.image.bin `
  --vocab-keep tools/vocabprune/idmaps/idmap_8192.json `
  --prompt "Hello, world!" --gen 8 --verify
```

Board flash, FPGA programming, UART, and `--verify` hardware steps: **NOT RUN**.
