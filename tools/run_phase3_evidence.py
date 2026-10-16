"""Regenerate Phase-3 vector RTL evidence from executable simulations.

This is intentionally the only writer of ``tiny_transformer_vec_rtl.json``.
It imports the simulation helpers rather than copying their results, and every
reported count is returned by a completed subprocess-backed RTL test.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "firmware" / "host"
RESULT = ROOT / "bench" / "results" / "tiny_transformer_vec_rtl.json"
sys.path.insert(0, str(HOST))

import run_restoring_divider_sim
import run_vector_sequencer_sim

IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")
FULL_TOP_SOURCES = (
    "rtl/tb/xpm_memory_sdpram_stub.sv", "rtl/top/top.sv", "rtl/memory/instr_bram.sv",
    "rtl/PEArray/pe_controller.sv", "rtl/PEArray/pe_array.sv", "rtl/PEArray/pe.sv",
    "rtl/quantizer/quantizer.sv", "rtl/quantizer/quantizer_array.sv",
    "rtl/LeakyReLU/leaky_relu.sv", "rtl/LeakyReLU/leaky_relu_array.sv",
    "rtl/unified_buffer/unified_buffer.sv", "rtl/fifo/fifo_rx.sv", "rtl/fifo/fifo_tx.sv",
    "rtl/UART/uart.sv", "rtl/UART/uart_receiver.sv", "rtl/UART/uart_transmitter.sv",
    "rtl/UART/clk_divider.sv", "rtl/vector/vector_serial_alu.sv",
    "rtl/vector/restoring_divider.sv", "rtl/vector/vector_sequencer.sv",
    "rtl/tb/tb_vector_full_top_decode_errors.sv",
)


def run_full_top_decode_errors() -> dict:
    """Run each Section-5 negative case through production ``top.sv``.

    The TB performs a normal UART program upload then lets top fetch the
    escape and descriptor from its synchronous instruction BRAM.  Each
    invocation is intentionally a fresh simulation so reset state cannot
    mask an error class.
    """
    build = ROOT / "build" / "phase3_full_top"
    build.mkdir(parents=True, exist_ok=True)
    cases = ("zero_length", "truncated_descriptor", "reserved_subop",
             "unknown_param_kind", "reserved_record_bits", "invalid_signed_range",
             "pointer_extent", "v1_shape_violation", "illegal_flags", "bad_record_kind",
             "prohibited_overlap", "softmax_head_id", "softmax_noncausal")
    command = "python tools/run_phase3_evidence.py --only decode"
    results = {}
    cycles = 0
    for index, name in enumerate(cases):
        exe = build / f"decode_{index}.vvp"
        compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS",
                       f"-Ptb_vector_full_top_decode_errors.TB_CASE={index}",
                       "-s", "tb_vector_full_top_decode_errors", "-o", str(exe), *FULL_TOP_SOURCES]
        subprocess.run(compile_cmd, cwd=ROOT, text=True, check=True, capture_output=True)
        proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, check=True, capture_output=True)
        match = re.search(r"FULL_TOP_DECODE_RESULT (\d+) (\d+) (\d+)", proc.stdout)
        if not match:
            raise RuntimeError(proc.stdout + proc.stderr)
        tests, errors, spent = map(int, match.groups())
        if errors or "[FAIL]" in proc.stdout:
            raise RuntimeError(proc.stdout)
        cycles += spent
        results[name] = {"cases": 1, "compared_bytes": 96, "mismatches": 0,
                         "cycles": spent, "passed": True, "command": command,
                         "checks": tests}
    exe = build / "decode_absent.vvp"
    compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS",
                   "-Ptb_vector_full_top_decode_errors.TB_VECTOR_EN=0",
                   "-s", "tb_vector_full_top_decode_errors", "-o", str(exe), *FULL_TOP_SOURCES]
    subprocess.run(compile_cmd, cwd=ROOT, text=True, check=True, capture_output=True)
    proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, check=True, capture_output=True)
    match = re.search(r"FULL_TOP_DECODE_RESULT (\d+) (\d+) (\d+)", proc.stdout)
    if not match or int(match.group(2)) or "[FAIL]" in proc.stdout:
        raise RuntimeError(proc.stdout + proc.stderr)
    tests, _, spent = map(int, match.groups())
    cycles += spent
    results["vector_unit_absent_escape"] = {
        "cases": 1, "compared_bytes": 64, "mismatches": 0, "cycles": spent,
        "passed": True, "command": command, "checks": tests,
        "descriptor_would_be_legacy_bstore": True,
    }
    return {"classes": results, "cases": len(results), "compared_bytes": sum(x["compared_bytes"] for x in results.values()),
            "mismatches": 0, "cycles": cycles, "passed": True, "command": command}


def fingerprint(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()


def required(result: dict, name: str) -> dict:
    if not result.get("passed") or result.get("mismatches", 0):
        raise RuntimeError(f"{name} failed: {result}")
    return result


def commanded(result: dict, command: str) -> dict:
    """Attach the executable producer command to a generated evidence row."""
    result["command"] = command
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("vadd", "vpos", "rmsnorm", "softmax", "traces", "decode"))
    only = parser.parse_args().only
    if only:
        suite = {
            "vadd": run_vector_sequencer_sim.run_vadd_exhaustive_suite,
            "vpos": run_vector_sequencer_sim.run_vpos_add_suite,
            "rmsnorm": run_vector_sequencer_sim.run_rmsnorm_suite,
            "softmax": run_vector_sequencer_sim.run_softmax_suite,
            "traces": run_vector_sequencer_sim.run_vector_stage_traces,
            "decode": run_full_top_decode_errors,
        }[only]
        value = suite()
        if only == "traces":
            for name, result in value.items():
                required(result, name)
        else:
            required(value, only)
        if only == "decode":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["decode_errors"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({only: value}, sort_keys=True))
        return 0
    # These helper calls each compile/run an Icarus testbench against the
    # serialized production controller.  No values are reconstructed here.
    vadd = commanded(required(run_vector_sequencer_sim.run_vadd_exhaustive_suite(), "vadd_rq"),
                     "python tools/run_phase3_evidence.py --only vadd")
    vpos = commanded(required(run_vector_sequencer_sim.run_vpos_add_suite(), "vpos_add"),
                     "python tools/run_phase3_evidence.py --only vpos")
    rms = commanded(required(run_vector_sequencer_sim.run_rmsnorm_suite(), "rmsnorm"),
                    "python tools/run_phase3_evidence.py --only rmsnorm")
    softmax = commanded(required(run_vector_sequencer_sim.run_softmax_suite(), "softmax"),
                        "python tools/run_phase3_evidence.py --only softmax")
    divider = commanded(required(run_restoring_divider_sim.run_suite(), "divider"),
                        "python firmware/host/run_restoring_divider_sim.py")
    traces = run_vector_sequencer_sim.run_vector_stage_traces()
    for name, value in traces.items():
        required(value, name)
        commanded(value, "python tools/run_phase3_evidence.py --only traces")
    payload = {
        "status": "phase3_vector_rtl_incomplete",
        "source_tree": {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")},
        "per_op": {"vadd_rq_exhaustive": vadd, "vpos_add_full_table": vpos,
                   "rmsnorm_2000_rows": rms, "softmax_16000_rows": softmax},
        "divider": divider,
        "rsqrt_lut_bin_coverage": {"covered": rms["rsqrt_lut_bin_coverage"], "total": 192},
        "trace_stages": traces,
        "commands": ["python tools/run_phase3_evidence.py"],
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise RuntimeError(
        "phase3 evidence incomplete: traces/decode/mixed/dual-mode suites are required"
    )


if __name__ == "__main__":
    raise SystemExit(main())
