import argparse
import json
import os
import sys
from typing import Any, Dict, List

from generate_fused_rtl_test_vectors import generate_vectors
from isa_simulator import simulate_mem_file
from run_rtl_fused_sim import _build_metrics, _find_simulator, _iverilog_run, _parse_sim_markers


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _decode_output_nibbles(fetch_bytes: List[int], count: int) -> List[int]:
    out: List[int] = []
    for byte in fetch_bytes:
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        out.append(lo - 16 if lo >= 8 else lo)
        out.append(hi - 16 if hi >= 8 else hi)
    return out[:count]


def _run_isa_cases(vectors: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    for case in vectors["cases"]:
        result = simulate_mem_file(case["program_mem"], array_size=vectors["array_size"])
        expected = case["expected_fetch_bytes"]
        fetch_match = result.fetch_bytes == expected
        cases.append({
            "name": case["name"],
            "program_mem": case["program_mem"],
            "program_words": case["program_words"],
            "expected_fetch_bytes": expected,
            "isa_fetch_bytes": result.fetch_bytes,
            "isa_output_int4": _decode_output_nibbles(result.fetch_bytes, len(case["expected_outputs"])),
            "expected_output_int4": case["expected_outputs"],
            "halted": result.halted,
            "pc": result.pc,
            "executed_ops": result.executed_ops,
            "isa_expected_bitmatch": bool(fetch_match and result.halted),
            "rtl_fetch_bytes": None,
            "isa_rtl_bitmatch": None,
        })
    return cases


def _run_rtl_metrics(repo_root: str, vectors: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _build_metrics(vectors)
    sim = _find_simulator()
    sim_log = ""
    status = "not_executed"
    if sim == "iverilog":
        passed, status, sim_log = _iverilog_run(repo_root)
        metrics["simulator_used"] = "iverilog"
        metrics["rtl_sim_executed"] = True
        metrics["rtl_sim_passed"] = bool(passed)
        metrics["fused_compressed_path_rtl_validated"] = bool(passed)
        _parse_sim_markers(sim_log, metrics)
    else:
        metrics["simulator_used"] = sim
        metrics["rtl_sim_executed"] = False
        metrics["rtl_sim_passed"] = False
        metrics["fused_compressed_path_rtl_validated"] = False
    metrics["rtl_status"] = status
    return metrics


def run_bitmatch(output_json: str, output_md: str) -> Dict[str, Any]:
    root = _repo_root()
    os.chdir(root)
    vectors = generate_vectors()
    cases = _run_isa_cases(vectors)
    rtl = _run_rtl_metrics(root, vectors)

    rtl_actual = rtl.get("actual_fetch_bytes") or {}
    for idx, case in enumerate(cases, start=1):
        key = f"case{idx}"
        rtl_bytes = rtl_actual.get(key)
        case["rtl_fetch_bytes"] = rtl_bytes
        case["isa_rtl_bitmatch"] = None if rtl_bytes is None else (case["isa_fetch_bytes"] == rtl_bytes)

    all_expected = all(case["isa_expected_bitmatch"] for case in cases)
    rtl_executed = bool(rtl.get("rtl_sim_executed"))
    all_rtl = bool(rtl_executed and rtl.get("rtl_sim_passed") and all(case["isa_rtl_bitmatch"] for case in cases))

    report = {
        "array_size": vectors["array_size"],
        "case_count": len(cases),
        "python_isa_simulator": "firmware/host/isa_simulator.py",
        "rtl_testbench": "rtl/tb/tb_fused_compressed_program.sv",
        "rtl_simulator_used": rtl.get("simulator_used"),
        "rtl_sim_executed": rtl_executed,
        "rtl_sim_passed": bool(rtl.get("rtl_sim_passed")),
        "all_isa_expected_bitmatch": bool(all_expected),
        "all_isa_rtl_bitmatch": bool(all_rtl),
        "cases": cases,
        "rtl_total_cycles": rtl.get("total_cycles"),
        "rtl_trace_log_path": rtl.get("trace_log_path"),
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        "# ISA Simulator / RTL Bitmatch Report",
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
    parser = argparse.ArgumentParser(description="Compare Python ISA simulator output against RTL simulation output")
    parser.add_argument("--output-json", default=os.path.join("build", "reports", "isa_rtl_bitmatch_report.json"))
    parser.add_argument("--output-md", default=os.path.join("build", "reports", "isa_rtl_bitmatch_report.md"))
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
