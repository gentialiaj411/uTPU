"""Regenerate Phase-3 vector RTL evidence from executable simulations.

This is intentionally the only writer of ``tiny_transformer_vec_rtl.json``.
It imports the simulation helpers rather than copying their results, and every
reported count is returned by a completed subprocess-backed RTL test.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "firmware" / "host"
RESULT = ROOT / "bench" / "results" / "tiny_transformer_vec_rtl.json"
sys.path.insert(0, str(HOST))

import run_restoring_divider_sim
import run_vector_sequencer_sim


def fingerprint(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()


def required(result: dict, name: str) -> dict:
    if not result.get("passed") or result.get("mismatches", 0):
        raise RuntimeError(f"{name} failed: {result}")
    return result


def main() -> int:
    # These helper calls each compile/run an Icarus testbench against the
    # serialized production controller.  No values are reconstructed here.
    vadd = required(run_vector_sequencer_sim.run_vadd_exhaustive_suite(), "vadd_rq")
    rms = required(run_vector_sequencer_sim.run_rmsnorm_suite(), "rmsnorm")

    # The position and softmax helpers are replaced by their suites as they
    # become available; fail closed until there is executable coverage at the
    # contractual count rather than perpetuating a static artifact claim.
    raise RuntimeError(
        "phase3 evidence incomplete: VPOS_ADD/softmax/traces/decode/mixed/dual-mode suites are required"
    )


if __name__ == "__main__":
    raise SystemExit(main())
