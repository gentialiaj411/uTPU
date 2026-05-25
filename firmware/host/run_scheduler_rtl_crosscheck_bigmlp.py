"""Phase 7 remediation P4.2 - wider scheduler RTL cross-check suite.

This suite runs the RTL scheduler TB against the board-fit shapes that
matter for the larger blocked-FC programs the board bitstream is meant
to flash. It reuses the same generator / simulator / RTL path as the
single-shape smoke cross-check, but runs multiple shapes one-by-one and
stores the per-shape results in a single artifact.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = REPO_ROOT / "firmware" / "host"
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from generate_scheduler_rtl_test_vectors import (  # noqa: E402
    build_case_vectors,
)

BUILD_DIR = REPO_ROOT / "build"
SIM_OUT_DIR = BUILD_DIR / "sim_iverilog"
TEST_VECTOR_DIR = BUILD_DIR / "test_vectors"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
OUTPUT_JSON = RESULTS_DIR / "scheduler_rtl_crosscheck_bigmlp.json"

DESIGN_FILES = [
    "rtl/tb/xpm_memory_sdpram_stub.sv",
    "rtl/top/top.sv",
    "rtl/memory/instr_bram.sv",
    "rtl/PEArray/pe_controller.sv",
    "rtl/PEArray/pe_array.sv",
    "rtl/PEArray/pe.sv",
    "rtl/quantizer/quantizer.sv",
    "rtl/quantizer/quantizer_array.sv",
    "rtl/LeakyReLU/leaky_relu.sv",
    "rtl/LeakyReLU/leaky_relu_array.sv",
    "rtl/unified_buffer/unified_buffer.sv",
    "rtl/fifo/fifo_rx.sv",
    "rtl/fifo/fifo_tx.sv",
    "rtl/UART/uart.sv",
    "rtl/UART/uart_receiver.sv",
    "rtl/UART/uart_transmitter.sv",
    "rtl/UART/clk_divider.sv",
]
TB_FILE = "rtl/tb/tb_scheduler_cycles.sv"

CASES = [
    {"out_features": 32, "in_features": 32, "tag": "bench_32x32"},
    {"out_features": 32, "in_features": 64, "tag": "bench_32x64"},
    {"out_features": 64, "in_features": 32, "tag": "bench_64x32"},
    {"out_features": 64, "in_features": 64, "tag": "bench_64x64"},
    {"out_features": 128, "in_features": 64, "tag": "bench_128x64"},
]

BOARD_LAYOUT = {
    "weight_addr": 256,
    "input_addr": 0,
    "result_addr": 320,
    "prog_depth": 8192,
    "array_size": 16,
}

_LOG_RE_RTL_NAIVE = re.compile(
    r"RTL_NAIVE_CYCLES=(\d+) \(sim=(\d+)\) NAIVE_FETCH_N=(\d+) \(sim=(\d+)\)"
)
_LOG_RE_RTL_SCHED = re.compile(
    r"RTL_SCHED_CYCLES=(\d+) \(sim=(\d+)\) SCHED_FETCH_N=(\d+) \(sim=(\d+)\)"
)
_LOG_RE_RTL_REDUCTION = re.compile(
    r"RTL_REDUCTION_PERMILLE=(\d+)\s+SIM_REDUCTION_PERMILLE=(\d+)\s+TOL=(\d+)\s+DIFF=(\d+)"
)
_LOG_RE_ADV_NAIVE = re.compile(r"ADVISORY: RTL_naive matches sim on (\d+) / (\d+) bytes")
_LOG_RE_ADV_SCHED = re.compile(r"ADVISORY: RTL_sched matches sim on (\d+) / (\d+) bytes")
_LOG_RE_DONE = re.compile(r"DONE tests=(\d+) errors=(\d+)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _resolve_iverilog() -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        (r"C:\iverilog\bin\iverilog.exe", r"C:\iverilog\bin\vvp.exe"),
        (
            r"C:\Program Files\Icarus Verilog\bin\iverilog.exe",
            r"C:\Program Files\Icarus Verilog\bin\vvp.exe",
        ),
    ]
    for iv, vv in candidates:
        if os.path.exists(iv) and os.path.exists(vv):
            return iv, vv
    iv_path = shutil.which("iverilog")
    vv_path = shutil.which("vvp")
    if iv_path and vv_path:
        return iv_path, vv_path
    return None, None


def _parse_log(text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {
        "raw_log": text,
        "tb_result_pass": "TB_RESULT: PASS" in text,
    }
    m = _LOG_RE_RTL_NAIVE.search(text)
    if m:
        parsed["rtl_naive_cycles"] = int(m.group(1))
        parsed["sim_naive_cycles_in_header"] = int(m.group(2))
        parsed["rtl_naive_fetch_n"] = int(m.group(3))
        parsed["sim_fetch_n_in_header"] = int(m.group(4))
    m = _LOG_RE_RTL_SCHED.search(text)
    if m:
        parsed["rtl_sched_cycles"] = int(m.group(1))
        parsed["sim_sched_cycles_in_header"] = int(m.group(2))
        parsed["rtl_sched_fetch_n"] = int(m.group(3))
    m = _LOG_RE_RTL_REDUCTION.search(text)
    if m:
        parsed["rtl_reduction_permille"] = int(m.group(1))
        parsed["sim_reduction_permille_in_header"] = int(m.group(2))
        parsed["tol_permille"] = int(m.group(3))
        parsed["diff_permille"] = int(m.group(4))
    m = _LOG_RE_ADV_NAIVE.search(text)
    if m:
        parsed["rtl_naive_bytes_matching_sim"] = int(m.group(1))
        parsed["rtl_naive_bytes_total"] = int(m.group(2))
    m = _LOG_RE_ADV_SCHED.search(text)
    if m:
        parsed["rtl_sched_bytes_matching_sim"] = int(m.group(1))
        parsed["rtl_sched_bytes_total"] = int(m.group(2))
    m = _LOG_RE_DONE.search(text)
    if m:
        parsed["tests"] = int(m.group(1))
        parsed["errors"] = int(m.group(2))
    return parsed


def _write_case_header(header_path: Path, case_files: Dict[str, str]) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(case_files["expected_svh"])
    shutil.copyfile(src, header_path)
    bytes_src = Path(case_files["expected_bytes_svh"])
    shutil.copyfile(bytes_src, header_path.with_name("scheduler_expected_bytes.svh"))


def _run_iverilog_flow(iv_bin: str, vv_bin: str, case_tag: str) -> Dict[str, object]:
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = SIM_OUT_DIR / "scheduler_rtl_crosscheck_bigmlp" / case_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_vvp = out_dir / "tb_scheduler_cycles.out"
    log_path = out_dir / "tb_scheduler_cycles.log"
    if out_vvp.exists():
        out_vvp.unlink()
    if log_path.exists():
        log_path.unlink()

    sources = [TB_FILE] + DESIGN_FILES
    compile_cmd = [
        iv_bin,
        "-g2012",
        "-DICARUS",
        "-DSCHED_TB_PROG_DEPTH=8192",
        "-o",
        str(out_vvp),
        *sources,
    ]
    compile_proc = subprocess.run(
        compile_cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if compile_proc.returncode != 0:
        return {
            "iverilog_compile_ok": False,
            "iverilog_compile_stderr": compile_proc.stdout,
            "log_path": str(log_path),
        }

    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(run_proc.stdout, encoding="utf-8")
    parsed = _parse_log(run_proc.stdout)
    return {
        "iverilog_compile_ok": True,
        "iverilog_compile_stderr": compile_proc.stdout,
        "iverilog_run_returncode": int(run_proc.returncode),
        "log_path": str(log_path),
        **parsed,
    }


def _methodology() -> Dict[str, object]:
    return {
        "api": "firmware/host/run_scheduler_rtl_crosscheck_bigmlp.py",
        "what_it_measures": (
            "RTL cycle reduction and RTL byte-exactness for the scheduler on "
            "board-fit blocked-FC shapes that exercise multi-out-block output "
            "addressing."
        ),
        "fit_criterion": (
            "Each case must satisfy RTL_scheduled_cycles < RTL_naive_cycles, "
            "|RTL_reduction_permille - sim_reduction_permille| <= 20, and "
            "RTL_naive fetch_bytes === RTL_scheduled fetch_bytes."
        ),
        "cases": CASES,
        "layout": BOARD_LAYOUT,
        "tolerance_permille": 20,
        "headline_assertions": [
            "RTL scheduled cycles strictly less than naive",
            "|RTL_reduction_permille - sim_reduction_permille| <= 20",
            "RTL_naive fetch_bytes === RTL_scheduled fetch_bytes",
            "RTL_naive fetch_n == sim fetch_n; RTL_sched fetch_n == sim fetch_n",
        ],
        "advisory_metrics": [
            "RTL_naive byte agreement with simulator (out of sim fetch_n)",
            "RTL_sched byte agreement with simulator (out of sim fetch_n)",
        ],
        "summary": (
            "Wider RTL cross-check for the board-fit suite. The suite uses the "
            "same scheduler / simulator / RTL pipeline as the single-shape "
            "smoke test, but it runs the larger shapes that are relevant to "
            "the board flash plan and stresses the multi-out-block output "
            "addressing path."
        ),
        "tools": {
            "naive_baseline": "firmware/host/lowering_blocked_fc_utpu.py (lower_blocked_fc_program_utpu)",
            "rtl_generator": "firmware/host/generate_scheduler_rtl_test_vectors.py::build_case_vectors",
            "scheduler": "firmware/host/scheduler_allocator.py (lower_blocked_fc_program_scheduled)",
            "simulator": "firmware/host/isa_simulator.py (1 cycle/op; STORE/BSTORE 2+N)",
            "testbench": "rtl/tb/tb_scheduler_cycles.sv",
        },
    }


def _run_case(iv_bin: Optional[str], vv_bin: Optional[str], case: Dict[str, object]) -> Dict[str, object]:
    tag = str(case["tag"])
    case_dir = TEST_VECTOR_DIR / "scheduler_rtl_crosscheck_bigmlp" / tag
    case_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"boardfit_{tag}"
    ref = build_case_vectors(
        out_features=int(case["out_features"]),
        in_features=int(case["in_features"]),
        array_size=BOARD_LAYOUT["array_size"],
        weight_addr=BOARD_LAYOUT["weight_addr"],
        input_addr=BOARD_LAYOUT["input_addr"],
        result_addr=BOARD_LAYOUT["result_addr"],
        prog_depth=BOARD_LAYOUT["prog_depth"],
        out_dir=case_dir,
        prefix=prefix,
    )
    header_copy = TEST_VECTOR_DIR / "scheduler_expected.svh"
    _write_case_header(header_copy, ref["paths"])

    rtl_result: Dict[str, object]
    if iv_bin and vv_bin:
        rtl_result = _run_iverilog_flow(iv_bin, vv_bin, tag)
    else:
        rtl_result = {
            "iverilog_compile_ok": False,
            "iverilog_compile_stderr": "iverilog unavailable",
            "status": "iverilog_unavailable",
        }

    case_report: Dict[str, object] = {
        "tag": tag,
        "shape": ref["shape"],
        "array_size": ref["array_size"],
        "weight_addr": ref["weight_addr"],
        "input_addr": ref["input_addr"],
        "result_addr": ref["result_addr"],
        "naive_words": ref["naive_words"],
        "sched_words": ref["sched_words"],
        "expected": {
            "shape": ref["shape"],
            "array_size": ref["array_size"],
            "weight_addr": ref["weight_addr"],
            "input_addr": ref["input_addr"],
            "result_addr": ref["result_addr"],
            "naive_words": ref["naive_words"],
            "sched_words": ref["sched_words"],
            "naive_cycles": ref["naive_cycles"],
            "sched_cycles": ref["sched_cycles"],
            "cycles_saved": ref["cycles_saved"],
            "reduction_permille": ref["reduction_permille"],
            "fetch_bytes_n": ref["fetch_bytes_n"],
            "expected_fetch_bytes": ref["expected_fetch_bytes"],
            "fetch_bytes_invariant_simulator": ref["fetch_bytes_invariant_simulator"],
        },
        "methodology": _methodology(),
        "tolerance_permille": 20,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "seed": ref["seed"],
        "paths": ref["paths"],
        "rtl_result": rtl_result,
    }

    if rtl_result.get("iverilog_compile_ok") and rtl_result.get("tb_result_pass"):
        case_report["headline"] = {
            "rtl_naive_cycles": rtl_result.get("rtl_naive_cycles"),
            "rtl_sched_cycles": rtl_result.get("rtl_sched_cycles"),
            "rtl_cycles_saved": None
            if rtl_result.get("rtl_naive_cycles") is None or rtl_result.get("rtl_sched_cycles") is None
            else int(rtl_result["rtl_naive_cycles"]) - int(rtl_result["rtl_sched_cycles"]),
            "rtl_reduction_permille": rtl_result.get("rtl_reduction_permille"),
            "sim_reduction_permille": rtl_result.get("sim_reduction_permille_in_header"),
            "diff_permille": rtl_result.get("diff_permille"),
            "tol_permille": rtl_result.get("tol_permille"),
            "scheduler_invariant_holds": bool(
                rtl_result.get("rtl_naive_bytes_matching_sim") == rtl_result.get("rtl_naive_bytes_total")
                and rtl_result.get("rtl_sched_bytes_matching_sim") == rtl_result.get("rtl_sched_bytes_total")
            ),
        }

    return case_report


def _aggregate(cases: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(cases)
    ok_cases = [c for c in cases if c.get("headline")]
    exact_cases = [
        c for c in cases
        if c.get("rtl_result", {}).get("rtl_naive_bytes_matching_sim") == c.get("rtl_result", {}).get("rtl_naive_bytes_total")
        and c.get("rtl_result", {}).get("rtl_sched_bytes_matching_sim") == c.get("rtl_result", {}).get("rtl_sched_bytes_total")
    ]
    return {
        "case_count": total,
        "ok_case_count": len(ok_cases),
        "all_cases_ok": len(ok_cases) == total,
        "all_cases_rtl_byte_exact": len(exact_cases) == total,
        "board_layout": BOARD_LAYOUT,
        "cases_fit_bram": {
            c["tag"]: bool(c["expected"]["naive_words"] <= BOARD_LAYOUT["prog_depth"] and c["expected"]["sched_words"] <= BOARD_LAYOUT["prog_depth"])
            for c in cases
        },
    }


def run_suite() -> Dict[str, object]:
    iv_bin, vv_bin = _resolve_iverilog()
    case_reports = [_run_case(iv_bin, vv_bin, case) for case in CASES]
    aggregate = _aggregate(case_reports)
    status = "ok"
    if not aggregate["all_cases_ok"] or not aggregate["all_cases_rtl_byte_exact"]:
        status = "failed"

    artifact = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "suite": "board_fit_bigmlp_v1",
        "status": status if iv_bin and vv_bin else "iverilog_unavailable",
        "tolerance_permille": 20,
        "methodology": _methodology(),
        "aggregate": aggregate,
        "cases": case_reports,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "iverilog": {
            "iverilog_bin": iv_bin,
            "vvp_bin": vv_bin,
        },
    }
    if status == "failed":
        artifact["status_reason"] = "one or more cases failed RTL/sim agreement or byte exactness"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[run_scheduler_rtl_crosscheck_bigmlp] status={artifact['status']} -> {OUTPUT_JSON}")
    return artifact


def main() -> int:
    run_suite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
