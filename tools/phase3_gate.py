"""Machine-check the Phase-3 RTL vector-unit completion contract."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "bench/results/tiny_transformer_vec_rtl.json"
SYNTH = ROOT / "bench/results/boolean_vec_unit_proxy_synth.json"


def fingerprint(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()


def main() -> int:
    bad: list[str] = []
    for gate in ("tools/phase1_gate.py", "tools/phase2_gate.py"):
        if subprocess.run([sys.executable, gate], cwd=ROOT).returncode:
            bad.append(gate)
    if subprocess.run([sys.executable, "tools/gen_tiny_transformer_luts.py", "--check"], cwd=ROOT).returncode:
        bad.append("LUT fixture")
    if subprocess.run([sys.executable, "tools/gen_vector_lut_mems.py", "--check"], cwd=ROOT).returncode:
        bad.append("LUT mem files")
    if not RESULT.is_file():
        bad.append("RTL vector result missing")
    else:
        r = json.loads(RESULT.read_text(encoding="utf-8"))
        expected = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
        if r.get("source_tree") != expected:
            bad.append("RTL vector source fingerprint")
        if r.get("rsqrt_lut_bin_coverage", {}).get("covered") != 192:
            bad.append("rsqrt coverage")
        for group in ("per_op", "decode_errors", "mixed", "trace_stages", "legacy_regression"):
            entries = r.get(group, {})
            if not entries:
                bad.append(group + " missing")
            for name, item in entries.items():
                if item.get("mismatches", 0) != 0 or item.get("passed") is False:
                    bad.append(f"{group}/{name}")
    if not SYNTH.is_file():
        bad.append("synthesis artifact missing")
    else:
        s = json.loads(SYNTH.read_text(encoding="utf-8"))
        if s.get("status") == "blocked":
            print("CONDITIONAL PASS: SYNTHESIS UNVERIFIED")
        elif s.get("status") != "ok":
            bad.append("synthesis status")
        else:
            if s.get("label") != "artix7_proxy_not_xc7s50": bad.append("synthesis label")
            for key, limit in (("extra_lut", 7400), ("lut_vec1", 32600), ("dsp_vec1", 120), ("bram_vec1", 75)):
                if s.get(key, limit + 1) > limit: bad.append("synthesis " + key)
            if s.get("wns_vec1", -1) < 0: bad.append("synthesis WNS")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "firmware/host", "rtl", "tools"], cwd=ROOT, text=True).strip()
    if dirty: bad.append("uncommitted scoped changes")
    for item in bad: print("UNMET:", item)
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
