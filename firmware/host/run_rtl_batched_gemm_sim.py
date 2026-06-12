import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

from generate_batched_gemm_rtl_vectors import generate_vectors


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


def _iverilog_run(repo_root: str) -> Tuple[bool, str]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found"
    build_dir = os.path.join(repo_root, "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, "tb_batched_gemm.out")
    srcs = [
        "rtl/tb/tb_batched_gemm.sv",
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
        return False, (c.stdout or "") + "\n" + (c.stderr or "")
    r = subprocess.run(run_cmd, cwd=repo_root, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    ok = (r.returncode == 0) and ("TB_RESULT: PASS" in (r.stdout or ""))
    return ok, (r.stdout or "") + "\n" + (r.stderr or "")


def _parse_perf_counter(log: str, label: str) -> Optional[int]:
    m = re.search(rf"{label}=(\d+)", log or "")
    if not m:
        return None
    return int(m.group(1))


def run_rtl_batched_gemm_sim(
    output_json: str,
    *,
    out_features: int = 16,
    in_features: int = 16,
    batch_size: int = 4,
    stem: Optional[str] = None,
) -> Dict[str, Any]:
    root = _repo_root()
    os.chdir(root)
    if stem is None:
        stem = f"batched_gemm_o{out_features}_i{in_features}_b{batch_size}"
    vectors = generate_vectors(
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        stem=stem,
        output_json=os.path.join("build", "test_vectors", f"{stem}.json"),
    )
    metrics: Dict[str, Any] = {
        "rtl_sim_executed": False,
        "rtl_sim_passed": False,
        "expected_fetch_bytes": vectors["expected_fetch_bytes"],
        "program_words": vectors["program_words"],
        "array_size": vectors["array_size"],
        "batch_size": vectors["batch_size"],
        "cfg": vectors["cfg"],
        "simulator_log": None,
        "perf_cycle_counter": None,
        "perf_busy_counter": None,
        "perf_program_count": None,
    }
    ok, log = _iverilog_run(root)
    metrics["simulator_log"] = log
    metrics["perf_cycle_counter"] = _parse_perf_counter(log, "PERF_CYCLE_COUNTER")
    metrics["perf_busy_counter"] = _parse_perf_counter(log, "PERF_BUSY_COUNTER")
    metrics["perf_program_count"] = _parse_perf_counter(log, "PERF_PROGRAM_COUNT")
    if "not found" not in log:
        metrics["rtl_sim_executed"] = True
        metrics["rtl_sim_passed"] = bool(ok)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batched GEMM RTL simulation validation")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()
    metrics = run_rtl_batched_gemm_sim(
        args.output_json,
        out_features=args.out_features,
        in_features=args.in_features,
        batch_size=args.batch_size,
        stem=args.stem,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
