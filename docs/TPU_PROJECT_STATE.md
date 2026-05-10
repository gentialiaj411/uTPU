# uTPU Project State (Canonical Handoff)

Last updated: 2026-05-10
Owner: Genti Aliaj

## Purpose
This is the canonical handoff file for LLM agents and human collaborators.
If code, RTL semantics, metrics, or validation status changes, this file must be updated in the same change.

## Current High-Level Status
- ARRAY_SIZE-aware software tiling is implemented (`ARRAY_SIZE=16` path).
- Blocked FC execution is implemented in host/compiler/runtime.
- Retargetable-compiler scaffold is introduced for blocked FC planning:
  - explicit `MemoryScope` (`REGISTER/SHARED/GLOBAL`)
  - `TargetDesc` abstraction
  - target-agnostic blocked-FC schedule builder
- Segmented blocked execution is implemented to handle BRAM limits.
- Compressed + fused full-inference program generation is implemented.
- Fused compressed full-inference fits instruction BRAM (1017 words <= 1024).
- RTL simulation for fused compressed flow passes with Icarus (`rtl_sim_passed=true`).
- Physical FPGA/UART board validation is still pending (no board access).

## Current Architecture (Practical)
1. PyTorch-trained quantized model is exported to int4 weights/scales.
2. `ProgramLoader` builds instruction programs:
   - Legacy 2x2 mode (preserved)
   - Blocked mode
   - Segmented blocked mode
   - Compressed mode
   - Fused compressed full-inference mode
3. ISA includes STORE/LOAD/RUN/FETCH/HALT and BSTORE.
4. RUN semantics include blocked-FC accumulation controls:
   - `acc_clear_en`
   - accumulate RUN (`compute=1, quantize=0, relu=0`)
   - finalize RUN (`compute=0, quantize=1, relu optional`)
5. RTL decode/control executes uploaded instruction programs.
6. Testbenches/scripts validate decode, accumulation, fetch correctness.

## Known Ground-Truth Metrics
Source reports:
- `build/reports/block_runtime_metrics.json`
- `build/reports/compressed_block_program_metrics.json`
- `build/reports/fused_full_inference_metrics.json`
- `build/reports/rtl_fused_sim_metrics.json`
- `build/reports/software_metrics.json`

### Runtime/Compilation Metrics
- `array_size`: 16
- Legacy host tile calls/inference: 515
- Segmented blocked phases: 5
- Fused full-inference phases: 1
- FC1 blocked words: 2701
- FC1 compressed words: 985
- FC2 blocked words: 217
- FC2 compressed words: 85
- Full compressed separate words: 1070
- Full fused words: 1017 (fits 1024)
- Words saved by fusion vs separate compressed: 53

### Accuracy/Correctness Metrics
- PyTorch accuracy (%): 90.0
- Tiled/runtime software accuracy (%): 90.0
- Accuracy delta (%): 0.0
- Legacy vs array_block (100-sample) max abs logit diff: 0.0
- RTL fused sim pass: true
- RTL case1 expected/actual bytes: [17, 245] / [17, 245]
- RTL case2 expected/actual bytes: [117, 119] / [117, 119]

Metric source note: Accuracy/equivalence metrics above are from `build/reports/block_runtime_metrics.json` generated on 2026-05-10 in this workspace run.

## Validation Commands (Current)
Run from repo root:

```powershell
python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md
python firmware/host/test_compressed_block_program.py
python firmware/host/test_fused_full_inference_program.py
python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md
```

## Key Files (Current)
- Host/compiler/runtime:
  - `firmware/host/compiler_abstractions.py`
  - `firmware/host/program_loader.py`
  - `firmware/host/isa_encoder.py`
  - `firmware/host/tiled_inference.py`
  - `firmware/host/block_runtime_analysis.py`
  - `firmware/host/test_compressed_block_program.py`
  - `firmware/host/test_fused_full_inference_program.py`
  - `firmware/host/run_rtl_fused_sim.py`
- RTL:
  - `rtl/top/top.sv`
  - `rtl/memory/instr_bram.sv`
  - `rtl/unified_buffer/unified_buffer.sv`
  - `rtl/PEArray/pe_controller.sv`
  - `rtl/tb/tb_fused_compressed_program.sv`
  - `rtl/tb/xpm_memory_sdpram_stub.sv`

## Open Risks / Remaining Blockers
- No physical FPGA/UART validation yet.
- Quantization-scale fidelity vs full training/deployment path should continue to be audited when changing arithmetic.
- Any ISA/RTL semantic change must preserve legacy 2x2 path unless intentionally deprecated.

## Update Protocol (Required for All Agents)
When changing code/RTL/tests/reports, update this file in the same commit:
1. Update `Last updated` date.
2. Update changed metrics with source report paths.
3. Add/remove validation commands if workflow changed.
4. Update blockers/risk section.
5. Add a short change note below.

## Change Notes
- 2026-05-10: Added canonical state file and synchronized with current validated metrics/reports.
- 2026-05-10: Added blocked-FC retargetable scaffold in `firmware/host/compiler_abstractions.py` with explicit memory scopes and target descriptors; `ProgramLoader.build_fc_layer_block_program()` now consumes the shared schedule builder without changing emitted ISA semantics.
- 2026-05-10: Re-ran `python firmware/host/block_runtime_analysis.py --num-samples 100 --output-json build/reports/block_runtime_metrics.json --output-md build/reports/block_runtime_report.md`; observed equivalence metrics 90.0% vs 90.0% and max abs logit diff 0.0.
- 2026-05-10: Post-change validation run results:
  - `python firmware/host/test_compressed_block_program.py` completed successfully; JSON output reports `decode_semantics_match=true`, FC1 compressed words=985, FC2 compressed words=85.
  - `python firmware/host/test_fused_full_inference_program.py` completed successfully; JSON output reports fused words=1017 (`fits_1024=true`), words saved vs separate compressed=53.
  - `python firmware/host/run_rtl_fused_sim.py --output-json build/reports/rtl_fused_sim_metrics.json --output-md build/reports/rtl_fused_sim_report.md` completed successfully with `rtl_sim_passed=true`.
