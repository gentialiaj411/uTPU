"""Narrow, auditable source fingerprints for transformer completion gates."""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = (
    "firmware/host/run_tiny_transformer_numerics.py", "firmware/host/run_tiny_transformer_divergence.py",
    "firmware/host/tiny_transformer_model.py", "firmware/host/tiny_transformer_reference.py",
    "firmware/host/tiny_transformer_export.py", "firmware/host/requantization.py",
    "firmware/host/fixtures/tiny_transformer_integer_luts.json", "firmware/host/fixtures/tiny_transformer_vocab.json",
    "firmware/host/fixtures/tiny_transformer_split.json", "tools/gen_tiny_transformer_luts.py",
)
PHASE2 = (
    "firmware/host/isa_encoder.py", "firmware/host/isa_simulator.py", "tools/run_tiny_transformer_vec_isa.py",
    "tools/isa_opcode_audit.py", "tools/phase_fingerprints.py", "firmware/host/test_tiny_transformer_vec_isa.py",
    "firmware/host/test_isa_simulator.py", "firmware/host/test_descriptor_isa_tile_gemm.py",
    "firmware/host/test_buffer_resident_lowering.py", "firmware/host/test_buffer_resident_deployment.py",
    "docs/transformer_on_utpu_design.md", "firmware/host/fixtures/tiny_transformer_integer_luts.json",
)

def fingerprint(paths: tuple[str, ...]) -> dict:
    return {"files": list(paths), "blobs": {p: subprocess.check_output(["git", "rev-parse", f"HEAD:{p}"], cwd=ROOT, text=True).strip() for p in paths}}

def valid(artifact: dict, paths: tuple[str, ...]) -> bool:
    return artifact.get("fingerprint_dependencies") == fingerprint(paths)
