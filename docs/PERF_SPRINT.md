# Performance sprint reproduction

This branch uses the existing integer golden semantics. See SPRINT_REPORT.md for final package acceptance; prepared scripts are not evidence that their RTL was adopted.

## Simulators

The shared decode simulator supports `--sim icarus` (default) and `--sim verilator`, including the `--sim=value` form. Fault subprocesses inherit `UTPU_SIM`. The current accelerated launcher targets Windows with WSL Ubuntu; it is not yet a native-Linux CI launcher.

Icarus tools use `C:\iverilog\bin`. Verilator tools default to the extracted installation under `../uTPU-research/research/data/tools/usr`; `UTPU_VERILATOR_TOOLS` overrides that directory. WSL needs g++, make and Python. Newly generated provenance requires NumPy and records Python/NumPy and simulator/compiler identities.

Verilator generated C++ and objects live in owned `/tmp/utpu-perf/*_obj` directories and are removed after a completed run or compile failure. Raw compiler output stays in `build/decode/*_compile.log`; generated testbench copies stay under `build/decode`. The cleanup prevents the WSL tmpfs from filling during the protocol variant sweep.

Testbenches without an explicit time declaration receive the same `1ns/1ps` timescale on both backends. UART task sampling is placed after the active clock edge; QSPI stateful random choices are assigned in explicit sequence before task calls. All scoreboards, assertions and mutation checks remain active.

## Useful commands

```powershell
python tools/decode/run_gemv_rtl.py --sim verilator
python tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --flash qspi --gen 8 --sim verilator --output build/decode/check_260k_qspi.json
python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --gen 8 --sim verilator --output build/decode/check_15m.json
python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --prompts 8 --gen 32 --sim verilator --timeout 5400 --output build/decode/check_15m_big.json
python -m pytest tools/decode/test_gate.py -q
python tools/decode_gate.py
```

The runner now honors the requested prompt count by extending its small default bank from distinct validation prefixes. It rejects prompt-plus-generation lengths that exceed the configured hardware context.

## Compact image proposal

If P4 is accepted in SPRINT_REPORT.md, GEMV records store the effective RNE multiplier in u16 plus the original u8 shift; embedding scale records retain u32 plus u8. The unchanged GEMV arithmetic receives the effective multiplier shifted left by sixteen bits. The packer constrains its range so that the existing signed multiplier interpretation stays valid.

The image ABI then differs from integration. Regenerate the packed image and matching wrappers together; mixing the old image with the new metadata decoder is invalid. The readonly preflight independently decodes every model tensor and compares the integer golden on original versus reconstructed metadata. RTL acceptance additionally requires the complete clean/mutation suites and the wrong-expansion fault.

```powershell
python tools/decode/pack_decode_image.py --model 15m
python tools/decode/gen_15m_wrap.py
python tools/decode/gen_qspi_wrap.py
```

No hardware programming is part of this sprint. `OWNER_RUN2.ps1` runs the pending implementation checks sequentially, logs to `build/owner_logs2`, stops on the first failure and runs the unchanged strict gate last. Any remaining fast regeneration requirements are listed in SPRINT_REPORT.md.
