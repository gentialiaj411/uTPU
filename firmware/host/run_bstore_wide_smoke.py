#!/usr/bin/env python3
"""Run BSTORE_WIDTH=8 iverilog smoke; emit bench/results/bstore_wide_smoke.json."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "bench" / "results" / "bstore_wide_smoke.json"
SIM_OUT = REPO / "build" / "sim_iverilog"

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
TB = "rtl/tb/tb_bstore_wide_smoke.sv"


def _tool(name: str) -> str | None:
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    return shutil.which(name)


def main() -> int:
    iverilog = _tool("iverilog")
    vvp = _tool("vvp")
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    out_bin = SIM_OUT / "tb_bstore_wide_smoke.out"
    report: dict = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "BSTORE_WIDTH": 8,
        "payload_words": 16,
    }
    if not iverilog or not vvp:
        report["status"] = "skipped_no_iverilog"
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("BSTORE wide smoke skipped: iverilog/vvp not found")
        return 0

    c = subprocess.run(
        [iverilog, "-g2012", "-DICARUS", "-o", str(out_bin), TB, *RTL_STUBS, *RTL_DESIGN],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if c.returncode != 0:
        report.update(
            {
                "status": "compile_fail",
                "stderr": (c.stderr or c.stdout)[-2000:],
            }
        )
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    r = subprocess.run([vvp, str(out_bin)], cwd=str(REPO), capture_output=True, text=True)
    text = (r.stdout or "") + (r.stderr or "")
    m_attr = re.search(r"ATTR_BSTORE=(\d+)", text)
    m_cpw = re.search(r"cyc_per_word=([0-9.]+)", text)
    passed = ("BSTORE_WIDE_SMOKE_PASS" in text) and r.returncode == 0
    report.update(
        {
            "status": "pass" if passed else "fail",
            "returncode": r.returncode,
            "attr_bstore": int(m_attr.group(1)) if m_attr else None,
            "cycles_per_payload_word": float(m_cpw.group(1)) if m_cpw else None,
            "stdout_tail": text[-2000:],
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "attr_bstore", "cycles_per_payload_word")}, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
