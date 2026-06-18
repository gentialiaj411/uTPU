from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from run_rtl_batched_gemm_sim import _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "pe_packed_pair_sim.json"
BUILD_DIR = REPO_ROOT / "build" / "sim_iverilog"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_pass_fail(log: str) -> Tuple[str, Optional[int]]:
    if "TB_RESULT: PASS" in log:
        match = re.search(r"PACKED_PAIR_VECTORS=(\d+)", log)
        return "PASS", int(match.group(1)) if match else None
    if "TB_RESULT: FAIL" in log:
        match = re.search(r"vectors=(\d+)", log)
        return "FAIL", int(match.group(1)) if match else None
    return "FAIL", None


def run_sim() -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        artifact = {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "skipped",
            "reason": "iverilog/vvp binaries not found",
            "result": None,
            "vector_count": None,
        }
        return False, "iverilog/vvp binaries not found", artifact

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = BUILD_DIR / "tb_pe_packed_pair.out"
    srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_pe_packed_pair.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_packed_pair.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
    ]
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", str(out_vvp)] + [str(s) for s in srcs]
    env = os.environ.copy()
    env["TMP"] = str(BUILD_DIR)
    env["TEMP"] = str(BUILD_DIR)
    env["TMPDIR"] = str(BUILD_DIR)

    compile_proc = subprocess.run(
        compile_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    if compile_proc.returncode != 0:
        log = (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")
        artifact = {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "compile_error",
            "reason": "iverilog compile failed",
            "result": "FAIL",
            "vector_count": None,
            "log_excerpt": log[-4000:],
        }
        return False, log, artifact

    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    log = (run_proc.stdout or "") + "\n" + (run_proc.stderr or "")
    result, vector_count = _parse_pass_fail(log)
    ok = run_proc.returncode == 0 and result == "PASS"
    artifact = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "status": "ran",
        "result": result,
        "vector_count": vector_count,
        "random_vectors": 1000,
        "column_depth": 8,
        "random_seed": "16'hACE1",
    }
    return ok, log, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pe_packed_pair iverilog self-check.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help="JSON artifact path (default: bench/results/pe_packed_pair_sim.json)",
    )
    args = parser.parse_args()

    ok, log, artifact = run_sim()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if artifact["status"] == "skipped":
        print(f"[run_pe_packed_pair_sim] skipped: {artifact['reason']}")
        print(f"[run_pe_packed_pair_sim] artifact: {args.output}")
        return 0

    print(log.rstrip())
    print(f"[run_pe_packed_pair_sim] artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
