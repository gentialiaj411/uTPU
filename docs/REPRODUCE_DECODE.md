# Reproducing the decode engine (R3/M6) from a fresh clone

Scope: RTL simulation (iverilog), out-of-context and board-top implementation (Vivado), and the gate. Nothing here has
been run on hardware. Run every command from the repository root. Artifacts land in `bench/results/`; re-running a runner
overwrites the tracked JSON there, so either restore with `git checkout -- bench/results` afterwards or pass `--output` to
a scratch path.

## 0. Tools and versions

| tool | used for | version | path |
|---|---|---|---|
| Python | all scripts | 3.11.9 (version used for the last local run of these steps) | on PATH |
| numpy | packers, goldens, runners | 2.4.6 (local run) | pip |
| torch (CPU) | llama2c reference, numerics export, `_linear_high` wrappers | 2.12.0+cpu (local run) | pip; not pinned in `requirements.txt` (GAP) |
| Icarus Verilog (`iverilog`, `vvp`) | every RTL simulation | not recorded in any artifact (GAP) | `C:\iverilog\bin\` (hard-coded in `tools/decode/sim.py`) |
| AMD Vivado | OOC and board-top implementation | 2025.2 (`tools/board/impl_top.py`) | `C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat` (hard-coded) |

The Windows paths are hard-coded in `tools/decode/sim.py` and `tools/board/impl_top.py`. Edit them there if the tools live
elsewhere. `requirements.txt` pins only the CUDA benchmark stack; it does not cover this flow.

## 1. Model sources (`models/`)

`models/` is not in the repository. Two sources:

1. **Scripted: stories260K, stories15M, tokenizer, eval slice.** `tools/llama2c/download.py` fetches every file pinned in
   `tools/llama2c/manifest.json` from karpathy/tinyllamas, karpathy/llama2.c and roneneldan/TinyStories, and checks SHA-256.
   Checked locally: `models/llama2c/stories15M.bin` = `cd590644...f49a`, matching the manifest.
   ```
   python tools/llama2c/download.py
   ```
2. **Scripted, but dependent on the calibration JSON: the 15M integer export.** `models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz`
   is produced by `firmware/host/numerics_v2/export_config.py` from `models/llama2c/stories15M.bin` and the committed
   `firmware/host/numerics_v2/calibration_15m.json`:
   ```
   python -m firmware.host.numerics_v2.export_config --name gptq_qkv_wo_w2_int8_g32 --mixed qkv=8,wo=8,w2=8
   ```
   The digest of the npz is `ecf950f4...fe03d`, recorded in `bench/results/numerics_v2_15m_50k.json` (`artifact_sha256`).
   The local npz matched that digest. The export was not re-run from scratch in this check, so bit-reproducibility of a fresh
   export (NumPy/zlib version dependence) is unverified (GAP).

   The 260K export is built in code, with no file needed beyond the checkpoint: `harness.load_runtime_260k()` reads
   `models/llama2c/stories260K/stories260K.bin` and `firmware/host/numerics_v2/calibration_260k.json`.

## 2. RTL LUT images (`rtl/decode/luts/*.mem`)

`*.mem` is git-ignored except under `firmware/host/numerics_v2/luts/`, so the RTL LUT images are generated, not checked in.
The source is the committed `firmware/host/numerics_v2/luts/luts.json`:
```
python tools/decode/gen_luts.py
```
This writes `rtl/decode/luts/{sigmoid_q15,exp_q15,rope_cos/sin_q15_h8,rope_cos/sin_q15_h48}.mem` and the 128-position RoPE
truncations used by the 15M build (`*_p128.mem`). The LUT generator, `firmware/host/numerics_v2/generate_luts.py`, takes
`--silu-input-scale` and `--silu-output-scale`, which are recorded in `luts.json` under `silu_i8.domain`. Re-running the
generator is not part of this bootstrap, and the committed `luts.json` is the source of truth here.

## 3. Packed 15M and 260K images

Both images come from `tools/decode/pack_decode_image.py`. `pack()` is the single packing function; `main()` writes
`build/decode/decode_image_<model>/{image.bin,params.json}`:
```
python tools/decode/pack_decode_image.py --model 260k   # default; output build/decode/decode_image_260k/
python tools/decode/pack_decode_image.py --model 15m    # build/decode/decode_image_15m/
```
Checks: the 260K image is 237,632 bytes; the 15M image is 13,073,472 bytes (`IMAGE_BYTES` in `params.json`, matching
`bench/results/decode_e2e_rtl_15m_kv4_p1.json` `image_bytes`). The 15M image must fit the 14,155,776-byte flash budget
(`rtl/board/boolean_core.sv`).

`tools/board/impl_top.py` checks `boolean_core` defaults against `build/decode/decode_image_15m/params.json`, so the 15M image
must exist before the board implementation step.

## 4. Golden captures

Each unit runner compares RTL output with golden vectors captured from the NumPy runtime. Capture first, then run the matching
runner. Some runners generate their own golden (noted below).

| golden | command | consumed by |
|---|---|---|
| GEMV (260K) | `python tools/decode/golden_gemv.py` (writes `build/decode/golden_gemv`) | `run_gemv_rtl.py` (default `--golden`) |
| GEMV (15M, K=288/768) | `python tools/decode/golden_gemv15.py` (writes `build/decode/golden_gemv15`) | `run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json` |
| activation quant | none; `run_act_quant_rtl.py` runs `golden_act_quant.py` itself | `run_act_quant_rtl.py` |
| RMSNorm | `python tools/decode/golden_rms.py` | `run_rms_rtl.py` |
| vec ALU | `python tools/decode/golden_vec.py` | `run_vec_rtl.py` |
| attention (260K) | `python tools/decode/golden_attn.py` | `run_attn_rtl.py` |
| attention (15M, HS=48, KV4, pos<=127) | `python tools/decode/golden_attn15.py` | `run_attn15_rtl.py` |
| QSPI | none; uses the flash model in `tools/decode/tb/qspi_flash_model.sv` | `run_qspi_rtl.py` |
| end to end | none; `run_decode_rtl.py` calls `golden()` itself | `run_decode_rtl.py` |

Golden directories are under `build/` and are not committed.

## 5. Unit runners (iverilog)

Run after the goldens above. Each writes its artifact to `bench/results/` by default.
```
python tools/decode/run_gemv_rtl.py
python tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json
python tools/decode/run_act_quant_rtl.py
python tools/decode/run_rms_rtl.py
python tools/decode/run_vec_rtl.py
python tools/decode/run_attn_rtl.py
python tools/decode/run_attn15_rtl.py
python tools/decode/run_qspi_rtl.py
python tools/decode/run_qspi_rtl.py --ddr    # EDh DDR reader (DDR=1) + flash model CR1 LC=11 -> decode_qspi_ddr_rtl.json; records measured bytes/core cycle (gate: >= 0.49)
```
Flash model and UART:
```
python tools/board/run_uart_rtl.py          # boolean_core + flash model: tokens, probed logits, QUAD enable, 87 checks
python tools/board/run_uart_rtl.py --ddr    # boolean_core DDR=1 ('V' = 2); flash starts at LC=00 and the host procedure must switch it to LC=11 -> boolean_uart_ddr_rtl.json
```
The `--ddr` paths need the DDR RTL (`qspi_reader` `DDR` parameter, `board_ctrl` version byte 2) and the flash model's `CR1_INIT` parameter. The parameter
names are constants at the top of `run_qspi_rtl.py`, `run_decode_rtl.py`, `run_uart_rtl.py` and `impl_top.py`.

## 6. End-to-end top_decode (slow)

`run_decode_rtl.py` compares every logit and token of every step with `RunQIntegerLlama`. Artifact names follow the tag
`decode_e2e_rtl_<model>_kv<bits>_p<period>[_qspi|_qspiddr]`.
```
# stories260K, ideal stream (5 prompts x 32 generated, the 169-step matrix)
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 1 --prompts 5 --gen 32   # decode_e2e_rtl_kv8_p1.json
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --period 1 --prompts 5 --gen 32   # decode_e2e_rtl_kv4_p1.json
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 2 --prompts 5 --gen 32   # decode_e2e_rtl_kv8_p2.json
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 4 --prompts 5 --gen 32   # decode_e2e_rtl_kv8_p4.json
# stories260K through qspi_reader and the flash model (1 prompt x 8)
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --flash qspi --prompts 1 --gen 8  # decode_e2e_rtl_kv8_p1_qspi.json
# stories260K through the DDR reader (EDh, flash model CR1 LC=11, 0.5 B/cycle)
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --flash qspi-ddr --prompts 1 --gen 8  # decode_e2e_rtl_kv8_p1_qspiddr.json
# stories15M, ideal stream, KV4 (1 prompt x 8). Wall time about 3 hours (artifact wall_seconds 10,302).
python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --prompts 1 --gen 8                 # decode_e2e_rtl_15m_kv4_p1.json
```
The 15M run packs its own image in memory from the npz, so it does not depend on step 3.
```
# stories15M through the DDR reader (1 prompt x 8; slow like the ideal 15M run)
python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi-ddr --prompts 1 --gen 8   # decode_e2e_rtl_15m_kv4_p1_qspiddr.json
```

## 7. Fault injection

Each unit's mutation build must be caught by its runner, and each fault must be detected or carry an equivalence argument:
```
python tools/decode/fault_inject.py <unit> [<unit> ...]     # results merge into bench/results/decode_fault_injection.json
```
Unit names are the keys of its table (`gemv`, `actq`, `rms`, `vec`, `attn`, `attn15`, `top`, `qspi`, `board`, `qspi_ddr`). `qspi_ddr` re-runs
`run_qspi_rtl.py --ddr` per mutant; its fault list in `fault_inject.py` is an empty placeholder until the DDR RTL change adds the EDh faults, and the gate
fails while it is empty.

## 8. Implementation (Vivado)

Out-of-context synthesis and place-and-route are recorded with `tools/decode/synth_ooc.py` (merges into
`bench/results/decode_synth_ooc.json`). The board top is implemented with:
```
python tools/board/impl_top.py      # requires build/decode/decode_image_15m/params.json; writes bench/results/boolean_top_impl.json
```
```
python tools/board/impl_top.py --ddr   # boolean_top DDR=1 -> bench/results/boolean_top_impl_ddr.json (reports in build/decode/synth/boolean_top_ddr)
python tools/decode/gen_qspi_wrap.py --ddr                                   # build/decode/top_decode_15m_qspi_ddr_wrap.sv
python tools/decode/synth_ooc.py qspi_reader_ddr qspi_reader rtl/decode/qspi_reader.sv -GDDR=1
python tools/decode/synth_ooc.py top_decode_15m_qspi_ddr top_decode_15m_qspi_ddr <the top_decode sources> rtl/decode/qspi_reader.sv build/decode/top_decode_15m_qspi_ddr_wrap.sv
```
`synth_ooc.py` takes `-GNAME=VALUE` to set a top-level generic (recorded as `generics` in the entry). `REGEN_ALL.ps1` runs all of these in order.
`impl_top.py` runs Vivado in batch mode on `constraints/boolean_xc7s50.xdc` and does not write a bitstream
(`bitstream_generated: false` in the artifact). `--reparse` re-reads the last run's reports without invoking Vivado.

## 9. Gate

```
python tools/decode_gate.py
```
Exit code 0 means every requirement listed in its docstring is met, based on the artifacts in `bench/results/`. The gate reads
the artifacts; it does not re-run the simulations. Re-running steps 5 to 8 is what makes its result current. The 15M
end-to-end artifact is one prompt of 8 tokens.

## Gaps (for the integration pass)

1. **`pack_decode_image.py` main() supported only 260K.** Fixed on this branch: `--model {260k,15m}` added, calling the existing
   `pack()`. The 260K output was hashed before and after (`image.bin` `4c3f7b450ca8a902fc214cd43f1ccaedfec3cfc52f849d9c0cfdb17c88d12d1e`, `params.json`
   `88e30cd8b9ef1e43084f6abf3adce6f6be907e0c995f8afdf1e8195ce54caaa7`): identical.
   The 15M image was verified to be 13,073,472 bytes.
2. **Model export not from scratch.** `gptq_qkv_wo_w2_int8_g32.npz` is produced by `export_config.py`, but its digest is recorded
   only in the artifact. A fresh export has not been checked for bit-identity.
3. **Tool versions not recorded.** No artifact records the iverilog or Vivado build. Only the install paths are known.
4. **torch and NumPy not pinned** for this flow. `requirements.txt` covers only the CUDA benchmarks.
5. **Hard-coded Windows tool paths** in `tools/decode/sim.py` and `tools/board/impl_top.py`.
6. **Generated LUT images are not committed** (`*.mem` is git-ignored outside `firmware/host/numerics_v2/luts/`). Step 2 is
   required before any RTL simulation.
7. **Runners overwrite tracked artifacts** in `bench/results/`. Use `--output` for scratch runs.
