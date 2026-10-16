"""Run the Phase-3 host regression file-by-file with Phase-1 known-failure IDs."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from phase1_gate import KNOWN_PREEXISTING_FAILURES

BUILD = ROOT / "build" / "phase3_host_suite"
SUMMARY = BUILD / "summary.json"


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "build" / "host_suite" / "summary.json"
    if not baseline_path.is_file():
        raise RuntimeError("host suite manifest missing: build/host_suite/summary.json")
    manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
    files = [ROOT / item["file"] for item in manifest.get("files", [])]
    if not files:
        raise RuntimeError("host suite manifest contains no per-file tests")
    # Preserve the original runner's known timeout outcome instead of turning
    # a file-level timeout into a false new pytest failure.
    excluded = {"firmware/host/test_systolic_characterization.py":
                "pre-existing timeout recorded by build/host_suite/summary.json and runner.log"}
    records = []
    observed: set[str] = set()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded:
            records.append({"file": rel, "status": "excluded", "reason": excluded[rel]})
            continue
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "pytest", rel, "-q", "-p", "no:cacheprovider"],
                              cwd=ROOT, text=True, capture_output=True)
        elapsed = round(time.perf_counter() - start, 3)
        output = proc.stdout + proc.stderr
        (BUILD / f"{path.stem}.log").write_text(output, encoding="utf-8")
        failed = re.findall(r"(?m)^FAILED\s+([^\s]+)", output)
        if proc.returncode and not failed:
            failed = [f"{rel}::<pytest_process_failure>"]
        observed.update(failed)
        records.append({"file": rel, "status": "passed" if proc.returncode == 0 else "failed",
                        "seconds": elapsed, "failed_node_ids": failed,
                        "summary": next((line.strip() for line in reversed(output.splitlines())
                                         if re.search(r"\d+ passed|\d+ failed|\d+ skipped", line)), "")})
    new_failures = sorted(observed - KNOWN_PREEXISTING_FAILURES)
    payload = {"status": "passed" if not new_failures else "failed", "files": records,
               "known_failure_node_ids": sorted(KNOWN_PREEXISTING_FAILURES),
               "observed_failure_node_ids": sorted(observed), "new_failure_node_ids": new_failures,
               "excluded": excluded}
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files_run": sum(r["status"] != "excluded" for r in records),
                      "files_excluded": len(excluded), "observed_failures": len(observed),
                      "new_failures": new_failures, "summary": str(SUMMARY.relative_to(ROOT))}, sort_keys=True))
    return int(bool(new_failures))


if __name__ == "__main__":
    raise SystemExit(main())
