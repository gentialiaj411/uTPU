# BATON
Owner: CODEX
Stage: IMPLEMENTATION_COMPLETE
Task: add TorchInductor oracle, Python ISA simulator/RTL bitmatch, and README front-page rewrite
Changed files:
- firmware/host/differential_test_harness.py
- firmware/host/test_differential_harness.py
- firmware/host/isa_simulator.py
- firmware/host/run_isa_rtl_bitmatch.py
- firmware/host/test_isa_simulator.py
- README.md
- docs/EVIDENCE.md
- docs/inspect_compiler_pipeline_demo.svg
Artifacts:
- build/reports/differential_test_report.json
- build/reports/isa_rtl_bitmatch_report.json
- build/reports/isa_rtl_bitmatch_report.md
- build/reports/compiler_introspection_tiny_mlp.json
Tests:
- python -m pytest firmware/host/test_isa_simulator.py -q
- python firmware/host/run_isa_rtl_bitmatch.py --output-json build/reports/isa_rtl_bitmatch_report.json --output-md build/reports/isa_rtl_bitmatch_report.md
- python -m pytest firmware/host/test_differential_harness.py firmware/host/test_isa_simulator.py -q
Result:
- Differential harness now includes a TorchInductor oracle entry; current Windows run skips it with `WinError 50`, while CUDA and uTPU emulation pass.
- Python ISA simulator matches expected fetch bytes and RTL fetch bytes for two compiled fused programs: `[17,245]` and `[117,119]`.
- README now leads with scoped claims, evidence links, architecture, and a terminal-preview SVG.
Notes:
- TorchInductor passing validation is not claim-safe until rerun on a supported Linux/WSL stack.
- uTPU differential harness still uses quantized runtime emulation; ISA/RTL bitmatch is simulation evidence, not board execution.
Next prompt:
- If claiming TorchInductor validation, rerun the differential harness on Linux/WSL with Inductor working.
- If strengthening ISA/RTL claim, generate more compiled programs and rerun `run_isa_rtl_bitmatch.py`.
Commit: DO_NOT_COMMIT
