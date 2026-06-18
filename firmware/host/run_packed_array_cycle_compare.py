from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_rtl_batched_gemm_sim import _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "packed_array_cycle_compare.json"
BUILD_DIR = REPO_ROOT / "build" / "sim_iverilog"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_cycle_lines(log: str) -> List[Dict[str, int]]:
    shapes: List[Dict[str, int]] = []
    for line in log.splitlines():
        m = re.match(
            r"CYCLE_SHAPE=(\d+) BASELINE_FIRST=(\d+) BASELINE_FULL=(\d+) "
            r"PACKED_FIRST=(\d+) PACKED_FULL=(\d+) STREAM=(\d+)",
            line,
        )
        if m:
            shapes.append(
                {
                    "array_size": int(m.group(1)),
                    "baseline_first": int(m.group(2)),
                    "baseline_full": int(m.group(3)),
                    "packed_first": int(m.group(4)),
                    "packed_full": int(m.group(5)),
                    "stream_cycles": int(m.group(6)),
                }
            )
    return shapes


def _build_shape_entry(row: Dict[str, int]) -> Dict[str, Any]:
    latency_delta_first = row["packed_first"] - row["baseline_first"]
    latency_delta_full = row["packed_full"] - row["baseline_full"]
    throughput_note = (
        "identical streaming schedule: packed matches baseline first/full capture cycles"
        if latency_delta_first == 0 and latency_delta_full == 0
        else (
            "PACKED THROUGHPUT REGRESSION: packed capture cycles exceed baseline "
            f"(first delta={latency_delta_first}, full delta={latency_delta_full})"
        )
    )
    return {
        "array_size": row["array_size"],
        "baseline_cycles": {
            "first_result": row["baseline_first"],
            "full_matrix": row["baseline_full"],
            "stream_total": row["stream_cycles"],
        },
        "packed_cycles": {
            "first_result": row["packed_first"],
            "full_matrix": row["packed_full"],
            "stream_total": row["stream_cycles"],
        },
        "latency_delta": {
            "first_result": latency_delta_first,
            "full_matrix": latency_delta_full,
        },
        "throughput_note": throughput_note,
    }


def _run_once(iv_bin: str, vv_bin: str, env: Dict[str, str]) -> Tuple[bool, str, List[Dict[str, int]]]:
    out_vvp = BUILD_DIR / "tb_pe_array_packed_cycles.out"
    srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_pe_array_packed_cycles.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
    ]
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
        log = (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")
        return False, log, []
    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    log = (run_proc.stdout or "") + "\n" + (run_proc.stderr or "")
    shapes = _parse_cycle_lines(log)
    ok = run_proc.returncode == 0 and "TB_RESULT: PASS" in log and len(shapes) == 2
    return ok, log, shapes


def run_sim() -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found", {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "skipped",
            "reason": "iverilog/vvp binaries not found",
            "result": None,
            "deterministic_across_runs": None,
            "scope_note": "iverilog-sim cycles, not silicon/synthesis",
        }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = str(BUILD_DIR)
    env["TEMP"] = str(BUILD_DIR)
    env["TMPDIR"] = str(BUILD_DIR)

    ok1, log1, shapes1 = _run_once(iv_bin, vv_bin, env)
    ok2, log2, shapes2 = _run_once(iv_bin, vv_bin, env)
    deterministic = ok1 and ok2 and shapes1 == shapes2
    log = log1 + "\n--- second run ---\n" + log2

    if not ok1:
        return False, log, {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "failed",
            "result": "FAIL",
            "deterministic_across_runs": False,
            "scope_note": "iverilog-sim cycles, not silicon/synthesis",
            "log_excerpt": log[-4000:],
        }

    configs = [_build_shape_entry(row) for row in shapes1]
    throughput_regression = any(
        "REGRESSION" in cfg["throughput_note"] for cfg in configs
    )
    artifact: Dict[str, Any] = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "status": "ran",
        "result": "PASS" if deterministic else "FAIL",
        "deterministic_across_runs": deterministic,
        "throughput_regression": throughput_regression,
        "configs": configs,
        "scope_note": "iverilog-sim cycles, not silicon/synthesis",
    }
    return deterministic, log, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run packed vs baseline cycle compare.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    ok, log, artifact = run_sim()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if artifact.get("status") == "skipped":
        print(f"[run_packed_array_cycle_compare] skipped: {artifact['reason']}")
        print(f"[run_packed_array_cycle_compare] artifact: {args.output}")
        return 0

    print(log.rstrip())
    print(f"[run_packed_array_cycle_compare] artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
