"""Fail-closed Phase-4 completion gate for the complete transformer."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISA = ROOT / "bench/results/tiny_transformer_program_isa.json"
RTL = ROOT / "bench/results/tiny_transformer_program_rtl.json"
SYNTH = ROOT / "bench/results/boolean_vec_unit_synth.json"

def run(cmd: list[str], bad: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode:
        bad.append("command failed: " + " ".join(cmd))

def rtl_blobs() -> dict[str, str]:
    paths = ["rtl/top/top.sv", "rtl/top/top_board.sv"]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "rtl/vector").glob("*.sv"))]
    return {p: subprocess.check_output(["git", "rev-parse", f"HEAD:{p}"], cwd=ROOT, text=True).strip() for p in paths}

def main() -> int:
    bad: list[str] = []
    py = sys.executable
    run([py, "tools/phase4a_gate.py"], bad)
    if not ISA.is_file():
        bad.append("ISA: artifact missing")
    else:
        isa = json.loads(ISA.read_text(encoding="utf-8"))
        if isa.get("status") != "passed" or isa.get("sequences") != 512: bad.append("ISA: too few sequences or failed status")
        if any(x.get("mismatches", 1) != 0 for x in isa.get("stages", {}).values()): bad.append("ISA: mismatches")
        m = isa.get("metrics", {})
        if abs(float(m.get("top1_accuracy", -1)) - 0.410097) > 1e-6 or abs(float(m.get("bits_per_char", -1)) - 2.904165) > 1e-6: bad.append("ISA: metric lock")
        if isa.get("program_words", 1 << 30) > 32768 or isa.get("weight_words", 1 << 30) > 27648 or isa.get("buffer_peak_words", 1 << 30) > 32768: bad.append("ISA: capacity")
    if not RTL.is_file():
        bad.append("RTL: artifact missing")
    else:
        rtl = json.loads(RTL.read_text(encoding="utf-8"))
        if rtl.get("status") != "passed": bad.append("RTL: failed status")
        if rtl.get("sequences", 0) < 8: bad.append("RTL: too few sequences")
        if rtl.get("mismatches", 1) != 0 or any(x.get("mismatches", 1) != 0 for x in rtl.get("stages", {}).values()): bad.append("RTL: mismatches")
        if not rtl.get("a5_weight_fill") or rtl.get("a5_sequences", 0) < 1: bad.append("RTL: A5 weight-fill proof")
    if not SYNTH.is_file():
        print("UNMET: synthesis: coordinator must run tools/run_boolean_synth.ps1")
    else:
        synth = json.loads(SYNTH.read_text(encoding="utf-8")); one = synth.get("modes", {}).get("1", {})
        if synth.get("rtl_head_blobs") != rtl_blobs() or one.get("wns_ns", -999) < 1.0: print("UNMET: synthesis: coordinator must run tools/run_boolean_synth.ps1")
    scoped = ["firmware/host", "rtl", "tools", "bench/results/tiny_transformer_program_isa.json", "bench/results/tiny_transformer_program_rtl.json"]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *scoped], cwd=ROOT, text=True).strip(): bad.append("artifacts: uncommitted scoped tree")
    for item in bad: print("UNMET:", item)
    return int(bool(bad))

if __name__ == "__main__": raise SystemExit(main())
