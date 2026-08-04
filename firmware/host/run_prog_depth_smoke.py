#!/usr/bin/env python3
"""Run parameterized PROG_DEPTH iverilog smoke across supported depths.

Emits bench/results/prog_depth_smoke.json. Exit nonzero if any depth fails.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "bench" / "results" / "prog_depth_smoke.json"
SIM_OUT = REPO / "build" / "sim_iverilog"
DEPTHS = [8192, 16384, 32768, 65536, 131072]

RTL_DESIGN = [
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
RTL_STUBS = ["rtl/tb/xpm_memory_sdpram_stub.sv"]
TB = "rtl/tb/tb_prog_depth_smoke.sv"


def _tool(name: str) -> str | None:
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    return shutil.which(name)


def run_depth(depth: int, *, iverilog: str, vvp: str) -> dict:
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    out_bin = SIM_OUT / f"tb_prog_depth_smoke_{depth}.out"
    cmd_c = [
        iverilog,
        "-g2012",
        "-DICARUS",
        f"-DTB_PROG_DEPTH={depth}",
        "-o",
        str(out_bin),
        TB,
        *RTL_STUBS,
        *RTL_DESIGN,
    ]
    c = subprocess.run(cmd_c, cwd=str(REPO), capture_output=True, text=True)
    if c.returncode != 0:
        return {
            "PROG_DEPTH": depth,
            "status": "compile_fail",
            "returncode": c.returncode,
            "stderr": (c.stderr or c.stdout)[-2000:],
        }
    r = subprocess.run([vvp, str(out_bin)], cwd=str(REPO), capture_output=True, text=True)
    text = (r.stdout or "") + (r.stderr or "")
    passed = ("PROG_DEPTH_SMOKE_PASS" in text) and r.returncode == 0
    return {
        "PROG_DEPTH": depth,
        "status": "pass" if passed else "fail",
        "returncode": r.returncode,
        "stdout_tail": text[-1500:],
    }


def main() -> int:
    iverilog = _tool("iverilog")
    vvp = _tool("vvp")
    if not iverilog or not vvp:
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "skipped_no_iverilog",
            "depths": DEPTHS,
            "points": [],
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PROG_DEPTH smoke skipped: iverilog/vvp not found")
        return 0

    points = [run_depth(d, iverilog=iverilog, vvp=vvp) for d in DEPTHS]
    all_pass = all(p["status"] == "pass" for p in points)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all_pass else "fail",
        "depths": DEPTHS,
        "points": points,
        "note": (
            "Functional regression for upload-path width bugs: PROG_DEPTH[15:0] "
            "truncation at 65536 and upload_count[PC_WIDTH-1:0] OOR at 131072."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "points": [
        {"PROG_DEPTH": p["PROG_DEPTH"], "status": p["status"]} for p in points
    ]}, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
