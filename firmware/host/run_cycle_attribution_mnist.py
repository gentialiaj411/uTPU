#!/usr/bin/env python3
"""Cycle attribution for the multi-layer MNIST fused program.

Settles whether result_fetch dominance is a single-GEMM harness artifact:
FC1 finalizes into the unified buffer without host FETCH; only final logits
are drained. Compare ATTR_RESULT_FETCH share vs cycle_attribution.json
flagship (64x64 B=48).

Requires existing build/test_vectors/mnist_utpu_expected.svh (from
run_mnist_utpu_demo.py). Compiles tb_mnist_utpu_program with -DMNIST_ATTR
-DMNIST_CASE_LIMIT=1 and UART_BAUD=1e8 (same on-chip scope as GEMM grid).
"""
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

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from run_rtl_batched_gemm_sim import _parse_perf_counter  # noqa: E402

OUT = REPO / "bench" / "results" / "cycle_attribution_mnist.json"
GEMM_ATTR = REPO / "bench" / "results" / "cycle_attribution.json"
SVH = REPO / "build" / "test_vectors" / "mnist_utpu_expected.svh"
ATTR_KEYS = [
    "fetch_decode",
    "store",
    "load",
    "compute",
    "writeback",
    "bstore",
    "ext_addr",
    "requant",
    "result_fetch",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def _pct(part: Optional[int], total: Optional[int]) -> Optional[float]:
    if part is None or total is None or total <= 0:
        return None
    return float(part) / float(total)


def _resolve_iverilog() -> Tuple[Optional[str], Optional[str]]:
    iv = shutil.which("iverilog")
    vv = shutil.which("vvp")
    if iv and vv:
        return iv, vv
    for iv_path, vv_path in [
        (r"C:\iverilog\bin\iverilog.exe", r"C:\iverilog\bin\vvp.exe"),
        (
            r"C:\Program Files\Icarus Verilog\bin\iverilog.exe",
            r"C:\Program Files\Icarus Verilog\bin\vvp.exe",
        ),
    ]:
        if os.path.exists(iv_path) and os.path.exists(vv_path):
            return iv_path, vv_path
    return None, None


def _run_mnist_attr() -> Tuple[bool, str]:
    iv, vv = _resolve_iverilog()
    if not iv or not vv:
        return False, "iverilog/vvp not found"
    build_dir = REPO / "build" / "rtl_sim"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_vvp = build_dir / "tb_mnist_attr.out"
    srcs = [
        "rtl/tb/tb_mnist_utpu_program.sv",
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
    env = os.environ.copy()
    env["TMP"] = str(build_dir)
    env["TEMP"] = str(build_dir)
    env["TMPDIR"] = str(build_dir)
    c = subprocess.run(
        [
            iv,
            "-g2012",
            "-DICARUS",
            "-DMNIST_ATTR",
            "-DMNIST_CASE_LIMIT=1",
            "-o",
            str(out_vvp),
        ]
        + [str(REPO / s) for s in srcs],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if c.returncode != 0:
        return False, (c.stdout or "") + "\n" + (c.stderr or "")
    r = subprocess.run(
        [vv, str(out_vvp)], cwd=REPO, env=env, capture_output=True, text=True
    )
    log = (r.stdout or "") + "\n" + (r.stderr or "")
    ok = r.returncode == 0 and "TB_RESULT: PASS" in log
    return ok, log


def _parse_svh_meta() -> Dict[str, Any]:
    text = SVH.read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    for key in ("CASE1_WORDS", "CASE1_FETCH_N", "TB_PROG_DEPTH", "TB_ARRAY_SIZE"):
        m = re.search(rf"`define {key} (\S+)", text)
        if m:
            raw = m.group(1).strip('"')
            try:
                meta[key.lower()] = int(raw)
            except ValueError:
                meta[key.lower()] = raw
    m = re.search(r'`define CASE1_NAME "([^"]+)"', text)
    if m:
        meta["case1_name"] = m.group(1)
    return meta


def main() -> int:
    if not SVH.exists():
        print(
            "MISSING vectors: run python firmware/host/run_mnist_utpu_demo.py first",
            flush=True,
        )
        return 2

    print("=== MNIST multi-layer cycle attribution (case1, fast-UART) ===", flush=True)
    ok, log = _run_mnist_attr()
    total = _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES")
    groups = {
        k: {
            "cycles": _parse_perf_counter(log, f"ATTR_{k.upper()}"),
            "pct_of_total_program_cycles": None,
        }
        for k in ATTR_KEYS
    }
    for k, g in groups.items():
        g["pct_of_total_program_cycles"] = _pct(g["cycles"], total)
    attributed = sum(int(g["cycles"] or 0) for g in groups.values())
    rf = groups["result_fetch"]["cycles"]
    rf_pct = groups["result_fetch"]["pct_of_total_program_cycles"]

    gemm_rf_pct = None
    gemm_total = None
    if GEMM_ATTR.exists():
        gemm = json.loads(GEMM_ATTR.read_text(encoding="utf-8"))
        flagship = next(
            (
                c
                for c in gemm.get("cases", [])
                if c.get("shape", {}).get("out_features") == 64
                and c.get("batch_size") == 48
            ),
            None,
        )
        if flagship:
            gemm_total = flagship.get("total_program_cycles")
            gemm_rf_pct = (
                flagship.get("groups", {})
                .get("result_fetch", {})
                .get("pct_of_total_program_cycles")
            )

    collapsed = bool(
        ok
        and rf_pct is not None
        and gemm_rf_pct is not None
        and rf_pct < 0.05
        and gemm_rf_pct > 0.20
    )
    verdict = (
        "result_fetch_harness_artifact_confirmed"
        if collapsed
        else (
            "inconclusive"
            if not ok
            else "result_fetch_still_material"
            if rf_pct is not None and rf_pct >= 0.05
            else "result_fetch_collapsed_no_gemm_baseline"
        )
    )

    bstore_c = groups["bstore"]["cycles"]
    compute_c = groups["compute"]["cycles"]
    amdahl_multilayer: Dict[str, Any] = {}
    if total is not None and bstore_c is not None and compute_c is not None:
        # Optimistic Part A: bstore payloads + compute survive; everything else removable.
        removable = int(total) - int(bstore_c) - int(compute_c)
        residual = int(total) - removable
        amdahl_multilayer = {
            "workload": "fused_mnist_case1_fast_uart_n16",
            "assumption": "Part A removes control micro-ops; bstore payloads and compute survive",
            "total_program_cycles": int(total),
            "bstore_cycles": int(bstore_c),
            "compute_cycles": int(compute_c),
            "removable_cycles": removable,
            "removable_fraction": removable / float(total),
            "residual_cycles": residual,
            "ceiling_x": float(total) / float(residual),
            "isolated_gemm_ceiling_x_do_not_use_here": 2.96,
        }

    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "methodology": {
            "program": "fused MNIST MLP (FC1 finalize_no_fetch → FC2 logits FETCH only)",
            "tb": "rtl/tb/tb_mnist_utpu_program.sv with -DMNIST_ATTR -DMNIST_CASE_LIMIT=1",
            "array_size": 16,
            "uart_baud_hz_testbench": 100000000,
            "uart_baud_hz_shipping_default": 115200,
            "measurement_scope": (
                "On-chip core cycles, fast-UART TB — same scope as cycle_attribution.json. "
                "Not board wall-clock."
            ),
            "svh_meta": _parse_svh_meta(),
        },
        "rtl_sim_passed": bool(ok),
        "total_program_cycles": total,
        "attr_sum_cycles": attributed,
        "attr_sum_vs_total_delta": (
            None if total is None else int(attributed) - int(total)
        ),
        "groups": groups,
        "comparison_to_isolated_gemm_64x64_b48": {
            "gemm_total_program_cycles": gemm_total,
            "gemm_result_fetch_pct": gemm_rf_pct,
            "mnist_result_fetch_pct": rf_pct,
            "mnist_result_fetch_cycles": rf,
            "fetch_bytes_expected_case1": _parse_svh_meta().get("case1_fetch_n"),
        },
        "verdict": verdict,
        "verdict_reason": (
            f"MNIST result_fetch={rf_pct:.4%}"
            if rf_pct is not None
            else "no result_fetch"
        )
        + (
            f" vs GEMM flagship {gemm_rf_pct:.4%}; "
            "inter-layer activations stay in buffer — drain is harness-only for single GEMM."
            if gemm_rf_pct is not None
            else ""
        ),
        "amdahl_part_a_multilayer": amdahl_multilayer,
        "log_tail": "\n".join((log or "").splitlines()[-40:]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "total_program_cycles", "groups")}, indent=2))
    print(f"wrote {OUT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
