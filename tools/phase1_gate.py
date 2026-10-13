"""Machine-check the Phase-1 transformer completion contract."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
HOST_SUMMARY = ROOT / "build" / "host_suite" / "summary.json"


def load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    failures: list[str] = []
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    divergence = load(RESULTS / "tiny_transformer_divergence.json")
    numerics = load(RESULTS / "tiny_transformer_numerics.json")
    if divergence is None:
        failures.append("divergence artifact missing")
    else:
        bad = [name for name, value in divergence.get("stages", {}).items() if float(value.get("cosine", -1.0)) < 0.98]
        if bad:
            failures.append("divergence cosine <0.98: " + ",".join(bad))
        if divergence.get("git_sha") != head:
            failures.append(f"divergence git_sha={divergence.get('git_sha')} HEAD={head}")
    if numerics is None:
        failures.append("numerics artifact missing")
    else:
        fv, iv = numerics.get("float_validation", {}), numerics.get("integer_validation", {})
        top_gap = abs(float(fv.get("top1_accuracy", -99)) - float(iv.get("top1_accuracy", 99)))
        bpc_gap = float(iv.get("bits_per_char", 99)) - float(fv.get("bits_per_char", -99))
        if top_gap > 0.03: failures.append(f"top1 gap={top_gap:.6f} >0.03")
        if bpc_gap > 0.3: failures.append(f"bpc gap={bpc_gap:.6f} >0.3")
        if int(numerics.get("training_steps", 0)) < 3000 and not numerics.get("plateau", False): failures.append("training_steps<3000 and plateau=false")
        if numerics.get("git_sha") != head: failures.append(f"numerics git_sha={numerics.get('git_sha')} HEAD={head}")
    lut = subprocess.run([sys.executable, "tools/gen_tiny_transformer_luts.py", "--check"], cwd=ROOT).returncode
    tests = subprocess.run([sys.executable, "-m", "pytest", "firmware/host/test_tiny_transformer_reference.py", "firmware/host/test_tiny_transformer_export.py", "-q", "-p", "no:cacheprovider"], cwd=ROOT).returncode
    if lut: failures.append("LUT generator check failed")
    if tests: failures.append("tiny_transformer pytest failed")
    summary = load(HOST_SUMMARY)
    if summary is None: failures.append("build/host_suite/summary.json missing")
    elif summary.get("failed") != 9 or summary.get("new_failures", []): failures.append(f"host suite failed={summary.get('failed')} new_failures={summary.get('new_failures')}")
    for failure in failures: print(f"UNMET: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
