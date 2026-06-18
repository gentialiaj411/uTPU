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
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "pe_array_packed_sim.json"
BUILD_DIR = REPO_ROOT / "build" / "sim_iverilog"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_pass_fail(log: str) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
    if "TB_RESULT: PASS" in log:
        gemm_match = re.search(r"PACKED_ARRAY_GEMMS=(\d+)", log)
        shapes_match = re.search(r"PACKED_ARRAY_SHAPES=([0-9,]+)", log)
        cases_match = re.search(r"PACKED_ARRAY_CASES=([a-zA-Z0-9_,]+)", log)
        return (
            "PASS",
            int(gemm_match.group(1)) if gemm_match else None,
            shapes_match.group(1) if shapes_match else None,
            cases_match.group(1) if cases_match else None,
        )
    if "TB_RESULT: FAIL" in log:
        return "FAIL", None, None, None
    return "FAIL", None, None, None


def _compile_and_run(iv_bin: str, vv_bin: str, out_vvp: Path, srcs: list[Path], env: Dict[str, str]) -> Tuple[int, str]:
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", str(out_vvp)] + [str(s) for s in srcs]
    compile_proc = subprocess.run(
        compile_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    if compile_proc.returncode != 0:
        return compile_proc.returncode, (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")
    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    return run_proc.returncode, (run_proc.stdout or "") + "\n" + (run_proc.stderr or "")


def run_sim() -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        artifact = {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "skipped",
            "reason": "iverilog/vvp binaries not found",
            "result": None,
            "shapes_tested": None,
            "gemm_count": None,
        }
        return False, "iverilog/vvp binaries not found", artifact

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = BUILD_DIR / "tb_pe_array_packed.out"
    guard_vvp = BUILD_DIR / "tb_pe_array_packed_odd_guard.out"
    srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
    ]
    guard_srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_pe_array_packed_odd_guard.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
    ]
    env = os.environ.copy()
    env["TMP"] = str(BUILD_DIR)
    env["TEMP"] = str(BUILD_DIR)
    env["TMPDIR"] = str(BUILD_DIR)

    rc, log = _compile_and_run(iv_bin, vv_bin, out_vvp, srcs, env)
    if rc != 0:
        artifact = {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "compile_error",
            "reason": "iverilog compile failed",
            "result": "FAIL",
            "shapes_tested": None,
            "gemm_count": None,
            "log_excerpt": log[-4000:],
        }
        return False, log, artifact

    result, gemm_count, shapes, cases = _parse_pass_fail(log)

    guard_rc, guard_log = _compile_and_run(iv_bin, vv_bin, guard_vvp, guard_srcs, env)
    guard_triggered = guard_rc != 0
    ok = rc == 0 and result == "PASS" and guard_triggered
    artifact = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "status": "ran",
        "result": "PASS" if ok else "FAIL",
        "shapes_tested": [8, 16, 32] if shapes is None else [int(x) for x in shapes.split(",")],
        "gemm_count": gemm_count,
        "cases_covered": [] if cases is None else cases.split(","),
        "random_gemms_per_shape": {"8": 200, "16": 200, "32": 64},
        "corner_cases_per_shape": 1,
        "odd_array_size_guard_triggered": bool(guard_triggered),
    }
    return ok, log + "\n" + guard_log, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pe_array_packed iverilog GEMM self-check.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help="JSON artifact path (default: bench/results/pe_array_packed_sim.json)",
    )
    args = parser.parse_args()

    ok, log, artifact = run_sim()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if artifact["status"] == "skipped":
        print(f"[run_pe_array_packed_sim] skipped: {artifact['reason']}")
        print(f"[run_pe_array_packed_sim] artifact: {args.output}")
        return 0

    print(log.rstrip())
    print(f"[run_pe_array_packed_sim] artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
