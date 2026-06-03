import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from generate_fused_residual_rtl_test_vectors import generate_vectors
from isa_encoder import IsaConfig
from isa_simulator import simulate_mem_file


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_iverilog_tools() -> Tuple[Optional[str], Optional[str]]:
    iv = shutil.which("iverilog")
    vv = shutil.which("vvp")
    if iv and vv:
        return iv, vv

    candidates = [
        (r"C:\iverilog\bin\iverilog.exe", r"C:\iverilog\bin\vvp.exe"),
        (r"C:\Program Files\Icarus Verilog\bin\iverilog.exe", r"C:\Program Files\Icarus Verilog\bin\vvp.exe"),
    ]
    for iv_path, vv_path in candidates:
        if os.path.exists(iv_path) and os.path.exists(vv_path):
            return iv_path, vv_path
    return None, None


def _find_simulator() -> Optional[str]:
    iv, vv = _resolve_iverilog_tools()
    if iv and vv:
        return "iverilog"
    for tool in ["verilator", "xsim"]:
        if shutil.which(tool):
            return tool
    return None


def _build_metrics(vectors: Dict[str, Any]) -> Dict[str, Any]:
    cases = vectors["cases"]
    return {
        "array_size": vectors["array_size"],
        "case_count": len(cases),
        "program_words": {f"case{idx}": case["program_words"] for idx, case in enumerate(cases, start=1)},
        "expected_outputs": {f"case{idx}": case["expected_outputs"] for idx, case in enumerate(cases, start=1)},
        "expected_fetch_bytes": {f"case{idx}": case["expected_fetch_bytes"] for idx, case in enumerate(cases, start=1)},
        "actual_fetch_bytes": {f"case{idx}": None for idx in range(1, len(cases) + 1)},
        "case_passed": {f"case{idx}": False for idx in range(1, len(cases) + 1)},
        "first_failure_stage": None,
        "first_failure_cycle": None,
        "first_failure_instruction": None,
        "total_cycles": None,
        "trace_log_path": "build/reports/rtl_fused_residual_trace.log",
        "simulator_used": None,
        "rtl_sim_executed": False,
        "rtl_sim_passed": False,
        "fused_residual_path_rtl_validated": False,
    }


def _parse_sim_markers(sim_log: str, metrics: Dict[str, Any]) -> None:
    for case_idx in range(1, int(metrics["case_count"]) + 1):
        bytes_match = re.search(rf"CASE{case_idx}_ACTUAL_BYTES=([0-9a-fA-F,]+)", sim_log)
        if bytes_match:
            parts = bytes_match.group(1).split(",")
            metrics["actual_fetch_bytes"][f"case{case_idx}"] = [int(p, 16) for p in parts if p]
        pass_match = re.search(rf"CASE{case_idx}_PASS=(\d+)", sim_log)
        if pass_match:
            metrics["case_passed"][f"case{case_idx}"] = (int(pass_match.group(1)) == 1)

    m = re.search(r"FIRST_FAILURE_STAGE=(\d+)", sim_log)
    if m:
        metrics["first_failure_stage"] = int(m.group(1))
    m = re.search(r"FIRST_FAILURE_CYCLE=(-?\d+)", sim_log)
    if m:
        metrics["first_failure_cycle"] = int(m.group(1))
    m = re.search(r"FIRST_FAILURE_INSTRUCTION=([0-9a-fA-F]{4})", sim_log)
    if m:
        metrics["first_failure_instruction"] = m.group(1)
    m = re.search(r"TOTAL_CYCLES=(\d+)", sim_log)
    if m:
        metrics["total_cycles"] = int(m.group(1))


def _iverilog_run(repo_root: str) -> Tuple[bool, str, str]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "not_executed", "iverilog/vvp binaries not found"
    build_dir = os.path.join(repo_root, "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, "tb_fused_residual_program.out")
    srcs = [
        "rtl/tb/tb_fused_residual_program.sv",
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
    srcs_abs = [os.path.join(repo_root, s) for s in srcs]
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", out_vvp] + srcs_abs
    run_cmd = [vv_bin, out_vvp]
    env = os.environ.copy()
    env["TMP"] = build_dir
    env["TEMP"] = build_dir
    env["TMPDIR"] = build_dir

    c = subprocess.run(compile_cmd, cwd=repo_root, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    if c.returncode != 0:
        return False, "compile_failed", (c.stdout or "") + "\n" + (c.stderr or "")
    r = subprocess.run(run_cmd, cwd=repo_root, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    ok = (r.returncode == 0) and ("TB_RESULT: PASS" in (r.stdout or ""))
    return ok, "executed", (r.stdout or "") + "\n" + (r.stderr or "")


def _run_isa_cases(vectors: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = IsaConfig(**vectors["cfg"])
    cases = []
    for case in vectors["cases"]:
        result = simulate_mem_file(
            case["program_mem"],
            array_size=vectors["array_size"],
            buffer_size=4096,
            cfg=cfg,
            accumulator_data_width=32,
        )
        fetch_match = result.fetch_bytes == case["expected_fetch_bytes"]
        cases.append({
            "name": case["name"],
            "program_mem": case["program_mem"],
            "program_words": case["program_words"],
            "expected_fetch_bytes": case["expected_fetch_bytes"],
            "isa_fetch_bytes": result.fetch_bytes,
            "halted": result.halted,
            "pc": result.pc,
            "executed_ops": result.executed_ops,
            "isa_expected_bitmatch": bool(fetch_match and result.halted),
            "rtl_fetch_bytes": None,
            "isa_rtl_bitmatch": None,
        })
    return cases


def run_bitmatch(output_json: str, output_md: str) -> Dict[str, Any]:
    root = _repo_root()
    os.chdir(root)
    vectors = generate_vectors()
    cases = _run_isa_cases(vectors)

    metrics = _build_metrics(vectors)
    sim = _find_simulator()
    sim_log = ""
    status = "not_executed"
    if sim == "iverilog":
        passed, status, sim_log = _iverilog_run(root)
        metrics["simulator_used"] = "iverilog"
        metrics["rtl_sim_executed"] = True
        metrics["rtl_sim_passed"] = bool(passed)
        metrics["fused_residual_path_rtl_validated"] = bool(passed)
        _parse_sim_markers(sim_log, metrics)
    else:
        metrics["simulator_used"] = sim
        metrics["rtl_sim_executed"] = False
        metrics["rtl_sim_passed"] = False
        metrics["fused_residual_path_rtl_validated"] = False

    rtl_actual = metrics.get("actual_fetch_bytes") or {}
    for idx, case in enumerate(cases, start=1):
        key = f"case{idx}"
        rtl_bytes = rtl_actual.get(key)
        case["rtl_fetch_bytes"] = rtl_bytes
        case["isa_rtl_bitmatch"] = None if rtl_bytes is None else (case["isa_fetch_bytes"] == rtl_bytes)

    all_expected = all(case["isa_expected_bitmatch"] for case in cases)
    rtl_executed = bool(metrics.get("rtl_sim_executed"))
    all_rtl = bool(rtl_executed and metrics.get("rtl_sim_passed") and all(case["isa_rtl_bitmatch"] for case in cases))

    report = {
        "array_size": vectors["array_size"],
        "case_count": len(cases),
        "python_isa_simulator": "firmware/host/isa_simulator.py",
        "rtl_testbench": "rtl/tb/tb_fused_residual_program.sv",
        "rtl_simulator_used": metrics.get("simulator_used"),
        "rtl_sim_executed": rtl_executed,
        "rtl_sim_passed": bool(metrics.get("rtl_sim_passed")),
        "all_isa_expected_bitmatch": bool(all_expected),
        "all_isa_rtl_bitmatch": bool(all_rtl),
        "cases": cases,
        "rtl_total_cycles": metrics.get("total_cycles"),
        "rtl_trace_log_path": metrics.get("trace_log_path"),
        "cfg": vectors.get("cfg"),
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        "# ISA Simulator / RTL Residual Bitmatch Report",
        "",
        f"- python_isa_simulator: {report['python_isa_simulator']}",
        f"- rtl_testbench: {report['rtl_testbench']}",
        f"- rtl_simulator_used: {report['rtl_simulator_used']}",
        f"- rtl_sim_executed: {report['rtl_sim_executed']}",
        f"- rtl_sim_passed: {report['rtl_sim_passed']}",
        f"- all_isa_expected_bitmatch: {report['all_isa_expected_bitmatch']}",
        f"- all_isa_rtl_bitmatch: {report['all_isa_rtl_bitmatch']}",
        f"- rtl_total_cycles: {report['rtl_total_cycles']}",
        "",
        "| Case | Words | Expected bytes | ISA bytes | RTL bytes | ISA==Expected | ISA==RTL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        md.append(
            f"| {case['name']} | {case['program_words']} | {case['expected_fetch_bytes']} | "
            f"{case['isa_fetch_bytes']} | {case['rtl_fetch_bytes']} | "
            f"{case['isa_expected_bitmatch']} | {case['isa_rtl_bitmatch']} |"
        )
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python ISA simulator output against residual RTL simulation output")
    parser.add_argument("--output-json", default=os.path.join("build", "reports", "isa_rtl_residual_bitmatch_report.json"))
    parser.add_argument("--output-md", default=os.path.join("build", "reports", "isa_rtl_residual_bitmatch_report.md"))
    args = parser.parse_args()
    report = run_bitmatch(args.output_json, args.output_md)
    print(json.dumps({
        "all_isa_expected_bitmatch": report["all_isa_expected_bitmatch"],
        "all_isa_rtl_bitmatch": report["all_isa_rtl_bitmatch"],
        "rtl_sim_executed": report["rtl_sim_executed"],
        "rtl_sim_passed": report["rtl_sim_passed"],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, indent=2))
    return 0 if report["all_isa_expected_bitmatch"] and report["all_isa_rtl_bitmatch"] else 1


if __name__ == "__main__":
    sys.exit(main())
