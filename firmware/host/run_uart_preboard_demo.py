from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from uart_replay import (
    DEMO_UART_BAUD,
    build_uart_replay_demo,
    parse_uart_captured_bytes,
    write_uart_replay_vectors,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "uart_preboard_roundtrip.json"


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


def _iverilog_run_uart_replay(repo_root: str) -> Tuple[bool, str]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found"
    build_dir = os.path.join(repo_root, "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, "tb_uart_replay.out")
    srcs = [
        "rtl/tb/tb_uart_replay.sv",
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
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", out_vvp] + [os.path.join(repo_root, s) for s in srcs]
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


def _parse_log_list(log: str, label: str) -> list[int]:
    match = re.search(rf"{label}=([0-9a-fA-F,]+)", log or "")
    if not match:
        return []
    return [int(part, 16) for part in match.group(1).split(",") if part]


def _parse_log_int(log: str, label: str) -> Optional[int]:
    match = re.search(rf"{label}=(\d+)", log or "")
    if not match:
        return None
    return int(match.group(1))


def build_artifact(output_json: Path = OUTPUT_JSON) -> Dict[str, Any]:
    demo = build_uart_replay_demo()
    vector_paths = write_uart_replay_vectors(demo, repo_root=REPO_ROOT)
    ok, log = _iverilog_run_uart_replay(str(REPO_ROOT))
    actual_uart = _parse_log_list(log, "UART_BYTES_ACTUAL")
    expected_uart = list(demo.expected_uart_bytes)
    decoded_actual = parse_uart_captured_bytes(
        actual_uart,
        out_features=8,
        batch_size=1,
        array_size=demo.array_size,
        cfg=demo.cfg,
    ) if actual_uart else []
    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "demo_program": {
            "name": demo.name,
            "scope_note": "Simulation-only UART replay of a lowered 8x8 INT8 single-tile blocked-FC demo program.",
            "program_words": int(demo.program_words),
            "prog_depth": int(demo.prog_depth),
            "fits_prog_depth": bool(demo.program_words <= demo.prog_depth),
            "array_size": int(demo.array_size),
            "compute_data_width": int(demo.cfg.compute_data_width),
            "address_width": int(demo.cfg.address_width),
            "buffer_size": int(demo.buffer_size),
            "uart_baud": int(DEMO_UART_BAUD),
            "weights": demo.weights.astype(int).tolist(),
            "activations": demo.activations.astype(int).tolist(),
            "requant_params": demo.requant_params.as_dict(),
        },
        "uart_roundtrip": {
            "upload_bytes_count": int(len(demo.upload_bytes)),
            "expected_uart_output_count": int(len(expected_uart)),
            "captured_uart_output_count": int(len(actual_uart)),
            "captured_uart_matches_isa_sim": bool(actual_uart == expected_uart),
            "expected_uart_output_bytes": expected_uart,
            "captured_uart_output_bytes": actual_uart,
            "decoded_expected_outputs": demo.expected_outputs.astype(int).tolist(),
            "decoded_captured_outputs": decoded_actual.astype(int).tolist() if hasattr(decoded_actual, "astype") else decoded_actual,
        },
        "rtl_simulation": {
            "rtl_sim_executed": "not found" not in log,
            "rtl_sim_passed": bool(ok),
            "no_x_quantizer_finalize": bool("No X on quantizer finalize outputs" in log),
            "no_x_uart_tx_path": bool("No X on UART TX line during capture" in log),
            "log_excerpt": "\n".join((log or "").splitlines()[-20:]),
        },
        "perf_counters": {
            "cycle_counter": _parse_log_int(log, "PERF_CYCLE_COUNTER"),
            "busy_counter": _parse_log_int(log, "PERF_BUSY_COUNTER"),
            "program_count": _parse_log_int(log, "PERF_PROGRAM_COUNT"),
        },
        "evidence_scaffold": {
            "run_log": "docs/evidence/fpga_onboard/RUN_LOG.md",
            "reproduce": "docs/evidence/fpga_onboard/REPRODUCE.md",
        },
        "on_silicon": {
            "status": "simulation",
            "scope_note": "This phase validates the host-device UART upload/capture/replay path in iverilog only. No board or silicon run was performed.",
        },
        "artifacts": vector_paths,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> int:
    artifact = build_artifact()
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "demo_program": artifact["demo_program"]["name"],
                "program_words": artifact["demo_program"]["program_words"],
                "prog_depth": artifact["demo_program"]["prog_depth"],
                "fits_prog_depth": artifact["demo_program"]["fits_prog_depth"],
                "captured_uart_matches_isa_sim": artifact["uart_roundtrip"]["captured_uart_matches_isa_sim"],
                "rtl_sim_passed": artifact["rtl_simulation"]["rtl_sim_passed"],
                "on_silicon_status": artifact["on_silicon"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
