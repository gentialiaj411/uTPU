"""Fail-closed Phase-4 complete-program gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISA = ROOT / "bench" / "results" / "tiny_transformer_program_isa.json"
RTL = ROOT / "bench" / "results" / "tiny_transformer_program_rtl.json"


def run(command: list[str], bad: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    if subprocess.run(command, cwd=ROOT).returncode:
        bad.append("command failed: " + " ".join(command))


def main() -> int:
    bad: list[str] = []
    run([sys.executable, "tools/phase4a_gate.py"], bad)
    run([sys.executable, "firmware/host/run_tiny_transformer_program_isa.py"], bad)
    if not ISA.is_file():
        bad.append("complete ISA artifact missing")
    else:
        isa = json.loads(ISA.read_text(encoding="utf-8"))
        if isa.get("status") != "passed" or isa.get("sequences", 0) < 512:
            bad.append("complete ISA execution")
        if isa.get("program_words", 1 << 30) > 32768 or isa.get("weight_words", 1 << 30) > 27648 or isa.get("buffer_peak_words", 1 << 30) > 32768:
            bad.append("program/weight/buffer capacity")
        if any(item.get("mismatches", 1) for item in isa.get("stages", {}).values()):
            bad.append("complete ISA stage bit-exactness")
        metrics = isa.get("metrics", {})
        if abs(float(metrics.get("top1_accuracy", -1)) - 0.410097) > 1e-6 or abs(float(metrics.get("bits_per_char", -1)) - 2.904165) > 1e-6:
            bad.append("complete ISA metric lock")
    if not RTL.is_file():
        bad.append("complete RTL artifact missing")
    else:
        rtl = json.loads(RTL.read_text(encoding="utf-8"))
        if rtl.get("status") != "passed" or rtl.get("sequences", 0) < 8 or any(item.get("mismatches", 1) for item in rtl.get("stages", {}).values()):
            bad.append("complete RTL bit-exactness")
        if not rtl.get("a5_weight_fill"):
            bad.append("A5 full-program proof")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "firmware/host", "rtl", "tools", "bench/results/tiny_transformer_program_isa.json", "bench/results/tiny_transformer_program_rtl.json"], cwd=ROOT, text=True).strip()
    if dirty:
        bad.append("uncommitted scoped changes")
    for item in bad:
        print("UNMET:", item)
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
