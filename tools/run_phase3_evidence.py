"""Regenerate Phase-3 vector RTL evidence from executable simulations.

This is intentionally the only writer of ``tiny_transformer_vec_rtl.json``.
It imports the simulation helpers rather than copying their results, and every
reported count is returned by a completed subprocess-backed RTL test.
"""
from __future__ import annotations

import json
import subprocess
import sys
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("vadd", "vpos", "rmsnorm", "softmax", "traces"))
    only = parser.parse_args().only
    if only:
        suite = {
            "vadd": run_vector_sequencer_sim.run_vadd_exhaustive_suite,
            "vpos": run_vector_sequencer_sim.run_vpos_add_suite,
            "rmsnorm": run_vector_sequencer_sim.run_rmsnorm_suite,
            "softmax": run_vector_sequencer_sim.run_softmax_suite,
            "traces": run_vector_sequencer_sim.run_vector_stage_traces,
        }[only]
        value = suite()
        if only == "traces":
            for name, result in value.items():
                required(result, name)
        else:
            required(value, only)
        print(json.dumps({only: value}, sort_keys=True))
        return 0
    # These helper calls each compile/run an Icarus testbench against the
    # serialized production controller.  No values are reconstructed here.
    vadd = required(run_vector_sequencer_sim.run_vadd_exhaustive_suite(), "vadd_rq")
    vpos = required(run_vector_sequencer_sim.run_vpos_add_suite(), "vpos_add")
    rms = required(run_vector_sequencer_sim.run_rmsnorm_suite(), "rmsnorm")
    softmax = required(run_vector_sequencer_sim.run_softmax_suite(), "softmax")
    divider = required(run_restoring_divider_sim.run_suite(), "divider")
    traces = run_vector_sequencer_sim.run_vector_stage_traces()
    for name, value in traces.items():
        required(value, name)
    payload = {
        "status": "phase3_vector_rtl_incomplete",
        "source_tree": {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")},
        "per_op": {"vadd_rq_exhaustive": vadd, "vpos_add_full_table": vpos,
                   "rmsnorm_2000_rows": rms, "softmax_16000_rows": softmax},
        "divider": divider,
        "rsqrt_lut_bin_coverage": {"covered": rms["rsqrt_lut_bin_coverage"], "total": 192},
        "trace_stages": traces,
        "commands": ["python tools/run_phase3_evidence.py"],
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise RuntimeError(
        "phase3 evidence incomplete: traces/decode/mixed/dual-mode suites are required"
    )


if __name__ == "__main__":
    raise SystemExit(main())
