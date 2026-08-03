import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

from generate_batched_gemm_rtl_vectors import generate_vectors
from isa_encoder import IsaConfig
from requantization import RequantParams


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


def _iverilog_run(
    repo_root: str,
    *,
    quantizer_lanes: Optional[int] = None,
    relu_lanes: Optional[int] = None,
    quantizer_pipe_depth: Optional[int] = None,
    out_vvp_name: str = "tb_batched_gemm.out",
) -> Tuple[bool, str]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found"
    build_dir = os.path.join(repo_root, "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, out_vvp_name)
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
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", out_vvp]
    if quantizer_lanes is not None:
        compile_cmd.append(f"-DBG_QUANTIZER_LANES={int(quantizer_lanes)}")
    if relu_lanes is not None:
        compile_cmd.append(f"-DBG_RELU_LANES={int(relu_lanes)}")
    if quantizer_pipe_depth is not None:
        compile_cmd.append(f"-DBG_QUANTIZER_PIPE_DEPTH={int(quantizer_pipe_depth)}")
    compile_cmd.extend(srcs_abs)
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
    hoist_tile_payloads: bool = False,
    cfg: Optional[IsaConfig] = None,
    accumulator_data_width: int = 32,
    requant_params: Optional[RequantParams] = None,
    quantizer_lanes: Optional[int] = None,
    relu_lanes: Optional[int] = None,
    quantizer_pipe_depth: Optional[int] = None,
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
        hoist_tile_payloads=hoist_tile_payloads,
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
        requant_params=requant_params,
    )
    metrics: Dict[str, Any] = {
        "rtl_sim_executed": False,
        "rtl_sim_passed": False,
        "expected_fetch_bytes": vectors["expected_fetch_bytes"],
        "program_words": vectors["program_words"],
        "array_size": vectors["array_size"],
        "batch_size": vectors["batch_size"],
        "cfg": vectors["cfg"],
        "hoist_tile_payloads": bool(vectors.get("hoist_tile_payloads", False)),
        "requant_params": vectors.get("requant_params"),
        "quantizer_lanes": quantizer_lanes,
        "relu_lanes": relu_lanes,
        "quantizer_pipe_depth": quantizer_pipe_depth,
        "simulator_log": None,
        "perf_cycle_counter": None,
        "perf_busy_counter": None,
        "perf_program_count": None,
        "compute_busy_cycles": None,
        "compute_span_cycles": None,
        "compute_span_duty_cycle": None,
        "finalize_requant_cycles": None,
    }
    vvp_name = "tb_batched_gemm.out"
    if quantizer_lanes is not None or quantizer_pipe_depth is not None:
        ql = "d" if quantizer_lanes is None else str(int(quantizer_lanes))
        pd = "d" if quantizer_pipe_depth is None else str(int(quantizer_pipe_depth))
        vvp_name = f"tb_batched_gemm_ql{ql}_pd{pd}.out"
    ok, log = _iverilog_run(
        root,
        quantizer_lanes=quantizer_lanes,
        relu_lanes=relu_lanes if relu_lanes is not None else quantizer_lanes,
        quantizer_pipe_depth=quantizer_pipe_depth,
        out_vvp_name=vvp_name,
    )
    metrics["simulator_log"] = log
    metrics["perf_cycle_counter"] = _parse_perf_counter(log, "PERF_CYCLE_COUNTER")
    metrics["perf_busy_counter"] = _parse_perf_counter(log, "PERF_BUSY_COUNTER")
    metrics["perf_program_count"] = _parse_perf_counter(log, "PERF_PROGRAM_COUNT")
    metrics["total_program_cycles"] = _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES")
    metrics["compute_busy_cycles"] = _parse_perf_counter(log, "COMPUTE_BUSY_CYCLES")
    metrics["compute_span_cycles"] = _parse_perf_counter(log, "COMPUTE_SPAN_CYCLES")
    metrics["finalize_requant_cycles"] = _parse_perf_counter(log, "FINALIZE_REQUANT_CYCLES")
    if metrics["compute_busy_cycles"] and metrics["compute_span_cycles"]:
        metrics["compute_span_duty_cycle"] = (
            float(metrics["compute_busy_cycles"]) / float(metrics["compute_span_cycles"])
        )
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
    parser.add_argument("--hoist-tile-payloads", action="store_true")
    parser.add_argument("--compute-data-width", type=int, default=4)
    parser.add_argument("--address-width", type=int, default=12)
    parser.add_argument("--accumulator-data-width", type=int, default=32)
    parser.add_argument("--requant-multiplier", type=int, default=None)
    parser.add_argument("--requant-right-shift", type=int, default=None)
    args = parser.parse_args()
    cfg = IsaConfig(address_width=args.address_width, compute_data_width=args.compute_data_width)
    requant_params = None
    if args.requant_multiplier is not None or args.requant_right_shift is not None:
        if args.requant_multiplier is None or args.requant_right_shift is None:
            raise SystemExit("both --requant-multiplier and --requant-right-shift are required together")
        requant_params = RequantParams(
            multiplier=int(args.requant_multiplier),
            right_shift=int(args.requant_right_shift),
            enable=True,
        )
    metrics = run_rtl_batched_gemm_sim(
        args.output_json,
        out_features=args.out_features,
        in_features=args.in_features,
        batch_size=args.batch_size,
        stem=args.stem,
        hoist_tile_payloads=bool(args.hoist_tile_payloads),
        cfg=cfg,
        accumulator_data_width=args.accumulator_data_width,
        requant_params=requant_params,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
