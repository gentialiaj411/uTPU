from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from run_rtl_batched_gemm_sim import _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "top_packed_smoke.json"
BUILD_DIR = REPO_ROOT / "build" / "sim_iverilog"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_sim() -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found", {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "status": "skipped",
            "reason": "iverilog/vvp binaries not found",
            "result": None,
            "wrapper_mode": "full_datapath_requant",
        }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = BUILD_DIR / "tb_top_packed_smoke.out"
    srcs = [
        REPO_ROOT / "rtl" / "tb" / "tb_top_packed_smoke.sv",
        REPO_ROOT / "rtl" / "top" / "top_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_controller_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_controller.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array_packed.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe_array.sv",
        REPO_ROOT / "rtl" / "PEArray" / "pe.sv",
        REPO_ROOT / "rtl" / "quantizer" / "quantizer_array.sv",
        REPO_ROOT / "rtl" / "quantizer" / "quantizer.sv",
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
            "wrapper_mode": "full_datapath_requant",
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
    wrapper_match = re.search(r"TOP_PACKED_WRAPPER=(\S+)", log)
    pass_all = run_proc.returncode == 0 and "TB_RESULT: PASS" in log
    artifact = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "status": "ran",
        "result": "PASS" if pass_all else "FAIL",
        "wrapper_mode": wrapper_match.group(1) if wrapper_match else "full_datapath_requant",
        "shapes_tested": [8, 16],
        "scope_note": "iverilog smoke: pe_controller+quantizer vs top_packed (accum+requant parity)",
    }
    return pass_all, log, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run top_packed smoke test.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    ok, log, artifact = run_sim()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if artifact.get("status") == "skipped":
        print(f"[run_top_packed_smoke] skipped: {artifact['reason']}")
        print(f"[run_top_packed_smoke] artifact: {args.output}")
        return 0

    print(log.rstrip())
    print(f"[run_top_packed_smoke] artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
