import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from generate_fused_rtl_test_vectors import generate_vectors


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


def _iverilog_run(repo_root: str) -> Tuple[bool, str, str]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "not_executed", "iverilog/vvp binaries not found"
    build_dir = os.path.join(repo_root, "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, "tb_fused_compressed_program.out")

    srcs = [
        "rtl/tb/tb_fused_compressed_program.sv",
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

    c = subprocess.run(compile_cmd, cwd=repo_root, capture_output=True, text=True)
    if c.returncode != 0:
        return False, "compile_failed", (c.stdout or "") + "\n" + (c.stderr or "")
    r = subprocess.run(run_cmd, cwd=repo_root, capture_output=True, text=True)
    ok = (r.returncode == 0) and ("TB_RESULT: PASS" in (r.stdout or ""))
    return ok, "executed", (r.stdout or "") + "\n" + (r.stderr or "")


def _build_metrics(vectors: Dict[str, Any]) -> Dict[str, Any]:
    c1 = vectors["cases"][0]
    c2 = vectors["cases"][1]
    return {
        "array_size": vectors["array_size"],
        "case_count": len(vectors["cases"]),
        "program_words": {"case1": c1["program_words"], "case2": c2["program_words"]},
        "bstore_count": {"case1": c1["bstore_count"], "case2": c2["bstore_count"]},
        "load_count": {"case1": c1["load_count"], "case2": c2["load_count"]},
        "run_count": {"case1": c1["run_count"], "case2": c2["run_count"]},
        "fetch_count": {"case1": c1["fetch_count"], "case2": c2["fetch_count"]},
        "halt_count": {"case1": c1["halt_count"], "case2": c2["halt_count"]},
        "expected_outputs": {"case1": c1["expected_outputs"], "case2": c2["expected_outputs"]},
        "expected_fetch_bytes": {
            "case1": c1["expected_fetch_bytes"],
            "case2": c2["expected_fetch_bytes"],
        },
        "actual_fetch_bytes": {"case1": None, "case2": None},
        "case1_passed": False,
        "case2_passed": False,
        "first_failure_stage": None,
        "first_failure_cycle": None,
        "first_failure_instruction": None,
        "total_cycles": None,
        "trace_log_path": "build/reports/rtl_fused_trace.log",
        "actual_outputs": None,
        "simulator_used": None,
        "rtl_sim_executed": False,
        "rtl_sim_passed": False,
        "fused_compressed_path_rtl_validated": False,
    }


def _parse_sim_markers(sim_log: str, metrics: Dict[str, Any]) -> None:
    def _parse_bytes(tag: str) -> Optional[List[int]]:
        m = re.search(rf"{tag}=([0-9a-fA-F]{{2}}),([0-9a-fA-F]{{2}})", sim_log)
        if not m:
            return None
        return [int(m.group(1), 16), int(m.group(2), 16)]

    c1_act = _parse_bytes("CASE1_ACTUAL_BYTES")
    c2_act = _parse_bytes("CASE2_ACTUAL_BYTES")
    if c1_act is not None:
        metrics["actual_fetch_bytes"]["case1"] = c1_act
    if c2_act is not None:
        metrics["actual_fetch_bytes"]["case2"] = c2_act

    m = re.search(r"CASE1_PASS=(\d+)", sim_log)
    if m:
        metrics["case1_passed"] = (int(m.group(1)) == 1)
    m = re.search(r"CASE2_PASS=(\d+)", sim_log)
    if m:
        metrics["case2_passed"] = (int(m.group(1)) == 1)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fused compressed RTL simulation validation")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)

    vectors = generate_vectors()
    metrics = _build_metrics(vectors)
    sim = _find_simulator()
    sim_log = ""
    status = "not_executed"

    if sim == "iverilog":
        passed, status, sim_log = _iverilog_run(root)
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

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md = [
        "# RTL Fused Compressed Simulation Report",
        "",
        f"- simulator_used: {metrics['simulator_used']}",
        f"- rtl_sim_executed: {metrics['rtl_sim_executed']}",
        f"- rtl_sim_passed: {metrics['rtl_sim_passed']}",
        f"- fused_compressed_path_rtl_validated: {metrics['fused_compressed_path_rtl_validated']}",
        f"- case1 program_words: {metrics['program_words']['case1']}",
        f"- case2 program_words: {metrics['program_words']['case2']}",
        f"- case1_passed: {metrics['case1_passed']}",
        f"- case2_passed: {metrics['case2_passed']}",
        f"- first_failure_stage: {metrics['first_failure_stage']}",
        f"- first_failure_cycle: {metrics['first_failure_cycle']}",
        f"- first_failure_instruction: {metrics['first_failure_instruction']}",
        f"- total_cycles: {metrics['total_cycles']}",
        f"- expected_fetch_bytes.case1: {metrics['expected_fetch_bytes']['case1']}",
        f"- actual_fetch_bytes.case1: {metrics['actual_fetch_bytes']['case1']}",
        f"- expected_fetch_bytes.case2: {metrics['expected_fetch_bytes']['case2']}",
        f"- actual_fetch_bytes.case2: {metrics['actual_fetch_bytes']['case2']}",
        f"- trace_log_path: {metrics['trace_log_path']}",
    ]
    if not metrics["rtl_sim_executed"]:
        md.extend([
            "",
            "## How To Run Locally",
            "```bash",
            "iverilog -g2012 -o build/rtl_sim/tb_fused_compressed_program.out \\",
            "  rtl/tb/tb_fused_compressed_program.sv rtl/top/top.sv rtl/memory/instr_bram.sv \\",
            "  rtl/PEArray/pe_controller.sv rtl/PEArray/pe_array.sv rtl/PEArray/pe.sv \\",
            "  rtl/quantizer/quantizer.sv rtl/quantizer/quantizer_array.sv \\",
            "  rtl/LeakyReLU/leaky_relu.sv rtl/LeakyReLU/leaky_relu_array.sv \\",
            "  rtl/unified_buffer/unified_buffer.sv rtl/fifo/fifo_rx.sv rtl/fifo/fifo_tx.sv \\",
            "  rtl/UART/uart.sv rtl/UART/uart_receiver.sv rtl/UART/uart_transmitter.sv rtl/UART/clk_divider.sv",
            "vvp build/rtl_sim/tb_fused_compressed_program.out",
            "```",
        ])
    if sim_log:
        md.extend(["", "## Simulator Log", "```text", sim_log.strip(), "```"])
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({
        "status": status,
        "simulator_used": metrics["simulator_used"],
        "rtl_sim_executed": metrics["rtl_sim_executed"],
        "rtl_sim_passed": metrics["rtl_sim_passed"],
        "metrics_json": args.output_json,
        "report_md": args.output_md,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
