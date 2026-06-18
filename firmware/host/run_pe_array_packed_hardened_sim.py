from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_rtl_batched_gemm_sim import _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "pe_array_packed_hardened.json"
BUILD_DIR = REPO_ROOT / "build" / "sim_iverilog"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shape_class_for(name: str) -> str:
    if name.startswith("odd_n_"):
        return "odd_logical_n"
    if name.startswith("rect_"):
        return "rectangular_gemm"
    if name.startswith("tile32_"):
        return "tile32"
    if name.startswith("batch"):
        return "batched_activations"
    return "other"


def _parse_classes(log: str) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    for line in log.splitlines():
        if line.startswith("HARDENED_CLASS "):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "PASS":
                classes.append({"name": parts[1], "status": "PASS"})
            elif len(parts) >= 3 and parts[2] == "CONSTRAINT":
                entry: Dict[str, Any] = {"name": parts[1], "status": "CONSTRAINT"}
                for token in parts[3:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        if k in {"row", "col", "cycle"}:
                            entry[k] = int(v)
                        elif k in {"base", "pack"}:
                            entry[k] = int(v)
                cause = next(
                    (
                        l.split(" ", 2)[2]
                        for l in log.splitlines()
                        if l.startswith(f"HARDENED_CAUSE {parts[1]}")
                    ),
                    None,
                )
                if cause:
                    entry["cause"] = cause
                classes.append(entry)
    return classes


def _aggregate_shape_classes(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for entry in classes:
        sc = _shape_class_for(entry["name"])
        if sc not in buckets:
            buckets[sc] = {
                "shape_class": sc,
                "status": "PASS",
                "case_count": 0,
                "pass_count": 0,
                "constraint_count": 0,
                "na_count": 0,
                "cases": [],
                "first_mismatch": None,
                "root_cause": None,
            }
        b = buckets[sc]
        b["case_count"] += 1
        b["cases"].append(entry["name"])
        status = entry.get("status", "PASS")
        if status == "PASS":
            b["pass_count"] += 1
        elif status == "CONSTRAINT":
            b["constraint_count"] += 1
            b["status"] = "CONSTRAINT"
            if b["first_mismatch"] is None:
                b["first_mismatch"] = {
                    k: entry[k]
                    for k in ("row", "col", "cycle", "base", "pack")
                    if k in entry
                }
                b["root_cause"] = entry.get("cause")
        elif status == "N-A":
            b["na_count"] += 1
            if b["status"] == "PASS":
                b["status"] = "N-A"
            b["root_cause"] = entry.get("reason", b["root_cause"])
        if b["constraint_count"] > 0:
            b["status"] = "CONSTRAINT"
    order = ["odd_logical_n", "rectangular_gemm", "tile32", "batched_activations", "other"]
    return [buckets[k] for k in order if k in buckets]


def run_sim() -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        artifact = {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "skipped",
            "reason": "iverilog/vvp binaries not found",
            "result": None,
            "classes": [],
            "shape_classes": [],
        }
        return False, "iverilog/vvp binaries not found", artifact

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = BUILD_DIR / "tb_pe_array_packed_hardened.out"
    srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_pe_array_packed_hardened.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
    ]
    env = os.environ.copy()
    env["TMP"] = str(BUILD_DIR)
    env["TEMP"] = str(BUILD_DIR)
    env["TMPDIR"] = str(BUILD_DIR)

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
        return False, log, {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "compile_error",
            "result": "FAIL",
            "classes": [],
            "shape_classes": [],
            "log_excerpt": log[-4000:],
        }

    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    log = (run_proc.stdout or "") + "\n" + (run_proc.stderr or "")
    classes = _parse_classes(log)
    shape_classes = _aggregate_shape_classes(classes)
    summary_match = re.search(r"HARDENED_SUMMARY cases=(\d+) fails=(\d+)", log)
    total_cases = int(summary_match.group(1)) if summary_match else len(classes)
    fail_cases = int(summary_match.group(2)) if summary_match else sum(1 for c in classes if c["status"] != "PASS")
    pass_all = run_proc.returncode == 0 and (
        "TB_RESULT: PASS" in log or "TB_RESULT: CONSTRAINT" in log
    )
    artifact = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "status": "ran",
        "result": "PASS" if pass_all else "FAIL",
        "total_cases": total_cases,
        "fail_cases": fail_cases,
        "classes": classes,
        "shape_classes": shape_classes,
        "scope_note": "iverilog simulation only; bit-exact vs baseline pe_array with explicit zero-padding",
    }
    return pass_all, log, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hardened pe_array_packed shape matrix.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    ok, log, artifact = run_sim()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if artifact["status"] == "skipped":
        print(f"[run_pe_array_packed_hardened_sim] skipped: {artifact['reason']}")
        print(f"[run_pe_array_packed_hardened_sim] artifact: {args.output}")
        return 0

    print(log.rstrip())
    print(f"[run_pe_array_packed_hardened_sim] artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
