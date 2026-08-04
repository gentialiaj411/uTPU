#!/usr/bin/env python3
"""Artix A7-100T PROG_DEPTH / BUFFER_SIZE capacity sweep (synthesis only).

Shipping datapath: ARRAY_SIZE=8 INT8, BUFFER_SIZE=4096 (PROG_DEPTH sweep),
MAX_BATCH_COUNT=48, QUANTIZER_PIPE_DEPTH=3, EXT_ADDR_EN=1.
Clock: best closing period for that config from timing_closure_sweep (20 ns).

Emits bench/results/prog_depth_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
HOST = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from run_timing_closure_sweep import (  # noqa: E402
    VIVADO,
    parse_timing,
    parse_util,
)

OUT = REPO / "bench" / "results" / "prog_depth_sweep.json"
REPORTS = REPO / "build" / "reports"

# Shipping close for mb48/pd3 (timing_closure_sweep.json).
CLOCK_PERIOD_NS = 20.0
MAX_BATCH = 48
PIPE_DEPTH = 3
ARRAY_SIZE = 8
COMPUTE_DATA_WIDTH = 8
ACCUMULATOR_DATA_WIDTH = 32
EXT_ADDR_EN = 1

PROG_DEPTHS = [8192, 16384, 32768, 65536, 131072]
BUFFER_SIZES = [4096, 16384, 65536]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def banking_check(buffer_size: int, array_size: int, compute_data_width: int) -> Dict[str, Any]:
    buffer_word = 16
    items = buffer_word // compute_data_width
    lanes = array_size * array_size
    banks_ok = items > 0 and (lanes % items == 0)
    banks = (lanes // items) if banks_ok else None
    depth_ok = bool(banks and banks > 0 and (buffer_size % banks == 0))
    bank_depth = (buffer_size // banks) if depth_ok and banks else None
    return {
        "buffer_size": buffer_size,
        "array_size": array_size,
        "compute_data_width": compute_data_width,
        "items_in_slot": items,
        "num_compute_lanes": lanes,
        "banks": banks,
        "bank_depth": bank_depth,
        "banks_integral": bool(banks_ok),
        "bank_depth_integral": bool(depth_ok),
        "ok": bool(banks_ok and depth_ok),
    }


def fetch_path_on_critical(timing: Dict[str, Any], timing_text: str) -> Dict[str, Any]:
    needles = (
        "pc",
        "bram_rd",
        "instr_bram",
        "FETCH_BRAM",
        "u_instr_bram",
        "bram_wr_addr",
        "upload_count",
    )
    blob = " ".join(
        [
            str(timing.get("critical_source") or ""),
            str(timing.get("critical_destination") or ""),
            timing_text,
        ]
    ).lower()
    hits = [n for n in needles if n.lower() in blob]
    # Narrow: only count if source/dest mention fetch/bram/pc, not whole file noise.
    src_dst = f"{timing.get('critical_source') or ''} {timing.get('critical_destination') or ''}".lower()
    src_dst_hits = [n for n in needles if n.lower() in src_dst]
    return {
        "fetch_related_in_source_or_dest": bool(src_dst_hits),
        "src_dst_hits": src_dst_hits,
        "critical_source": timing.get("critical_source"),
        "critical_destination": timing.get("critical_destination"),
        "note": (
            "True means PC/instr-BRAM appears on the reported critical endpoints; "
            "False means some other path binds (often PE/compute/load)."
        ),
    }


def lutram_spill_hint(util_text: str) -> Dict[str, Any]:
    # Vivado util often has "LUT as Memory" / Distributed RAM lines.
    m = re.search(r"\|\s*LUT as Memory\s*\|\s*(\d+)\s*\|", util_text)
    lut_mem = int(m.group(1)) if m else None
    m2 = re.search(r"\|\s*Block RAM Tile\s*\|\s*(\d+)\s*\|", util_text)
    bram = int(m2.group(1)) if m2 else None
    return {
        "lut_as_memory": lut_mem,
        "bram_36k_used": bram,
        "possible_lutram_spill": bool(lut_mem is not None and lut_mem > 500),
    }


def run_vivado(
    *,
    prog_depth: int,
    buffer_size: int,
    report_prefix: str,
) -> int:
    log_dir = REPO / "build" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    proj = (
        f"build/vivado_arty_a7_prog{prog_depth}_buf{buffer_size}"
        f"_mb{MAX_BATCH}_clk{CLOCK_PERIOD_NS:g}_pd{PIPE_DEPTH}"
    )
    args = [
        str(VIVADO),
        "-mode",
        "batch",
        "-source",
        "program_arty_a7_revE.tcl",
        "-log",
        str(log_dir / f"{report_prefix}.vlog"),
        "-journal",
        str(log_dir / f"{report_prefix}.jou"),
        "-tclargs",
        "proj_dir",
        proj,
        "proj_name",
        f"uTPU_prog{prog_depth}_buf{buffer_size}",
        "PROG_DEPTH",
        str(prog_depth),
        "COMPUTE_DATA_WIDTH",
        str(COMPUTE_DATA_WIDTH),
        "ACCUMULATOR_DATA_WIDTH",
        str(ACCUMULATOR_DATA_WIDTH),
        "ARRAY_SIZE",
        str(ARRAY_SIZE),
        "BUFFER_SIZE",
        str(buffer_size),
        "EXT_ADDR_EN",
        str(EXT_ADDR_EN),
        "MAX_BATCH_COUNT",
        str(MAX_BATCH),
        "QUANTIZER_LANES",
        str(ARRAY_SIZE),
        "RELU_LANES",
        str(ARRAY_SIZE),
        "QUANTIZER_PIPE_DEPTH",
        str(PIPE_DEPTH),
        "clock_period",
        str(CLOCK_PERIOD_NS),
        "report_prefix",
        report_prefix,
        "do_program",
        "0",
    ]
    print("RUN", report_prefix, flush=True)
    # Redirect Vivado stderr so PowerShell does not abort the parent on
    # Vivado's "ERROR: Failed runs" messages (which can be non-fatal race noise).
    log_path = log_dir / f"{report_prefix}.vlog"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        return subprocess.call(args, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)


def collect_point(report_prefix: str, *, prog_depth: int, buffer_size: int) -> Dict[str, Any]:
    timing_path = REPORTS / f"{report_prefix}_timing_summary.rpt"
    util_path = REPORTS / f"{report_prefix}_utilization.rpt"
    point: Dict[str, Any] = {
        "report_prefix": report_prefix,
        "PROG_DEPTH": prog_depth,
        "BUFFER_SIZE": buffer_size,
        "PC_WIDTH": int(math.ceil(math.log2(prog_depth))) if prog_depth > 0 else None,
        "ARRAY_SIZE": ARRAY_SIZE,
        "COMPUTE_DATA_WIDTH": COMPUTE_DATA_WIDTH,
        "MAX_BATCH_COUNT": MAX_BATCH,
        "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
        "clock_period_ns": CLOCK_PERIOD_NS,
        "banking": banking_check(buffer_size, ARRAY_SIZE, COMPUTE_DATA_WIDTH),
    }
    if not timing_path.exists() or not util_path.exists():
        # Prefer a concrete synth-fail diagnosis when the Vivado log exists.
        vlog = REPO / "build" / "logs" / f"{report_prefix}.vlog"
        if vlog.exists():
            vtxt = vlog.read_text(encoding="utf-8", errors="replace")
            if re.search(r"ERROR: \[Synth", vtxt) or "Synthesis failed" in vtxt:
                m = re.search(r"ERROR: \[Synth[^\]]*\] ([^\n]+)", vtxt)
                point["status"] = "failed_synth"
                point["failure_detail"] = {
                    "reason": "synth_error",
                    "first_synth_error": m.group(1).strip() if m else None,
                    "vlog": str(vlog.relative_to(REPO)).replace("\\", "/"),
                }
                return point
        point["status"] = "missing_reports"
        return point
    timing_text = timing_path.read_text(encoding="utf-8", errors="replace")
    util_text = util_path.read_text(encoding="utf-8", errors="replace")
    timing = parse_timing(timing_path)
    util = parse_util(util_path)
    point["timing"] = timing
    point["utilization"] = util
    point["fetch_path"] = fetch_path_on_critical(timing, timing_text)
    point["memory_style"] = lutram_spill_hint(util_text)
    m36 = re.search(r"\|\s*RAMB36E1 only\s*\|\s*(\d+)\s*\|", util_text)
    ramb36 = int(m36.group(1)) if m36 else None
    point["ramb36e1_only"] = ramb36
    # Expected instr BRAM: PROG_DEPTH*16 bits / 36 Kib per RAMB36.
    expected_instr_ramb36 = max(1, (int(prog_depth) * 16 + 36863) // 36864)
    point["expected_instr_ramb36_min"] = expected_instr_ramb36
    point["status"] = "closed" if timing.get("all_paths_met") else "failed_timing"
    # Guardrail: RAMB36 must cover PROG_DEPTH×16b (~36 Kib/RAMB36). Observed bad
    # point: PROG_DEPTH=65536 timed closed with only 2 RAMB36 (need >=29) and LUT
    # collapsed ~55k→11k — not a capacity proof.
    lut_used = util.get("lut_used")
    design_collapsed = lut_used is not None and lut_used < 25000 and int(prog_depth) >= 32768
    if (
        ramb36 is not None
        and int(prog_depth) >= 65536
        and (ramb36 < expected_instr_ramb36 or design_collapsed)
    ):
        point["status"] = "failed_instr_bram_undercount"
        point["failure_detail"] = {
            "reason": "instr_bram_block_ram_undercount",
            "ramb36e1_only": ramb36,
            "expected_instr_ramb36_min": expected_instr_ramb36,
            "lut_used": lut_used,
            "design_collapsed_lut": design_collapsed,
            "note": (
                "Timing may meet, but post-route RAMB36 is far below storage "
                "required for PROG_DEPTH×16b (and/or LUT collapsed vs 32k close). "
                "Invalid as a capacity proof. Synth log still shows 64Kx16 "
                "inference intent; final mapped count does not."
            ),
        }
    if util.get("bram_36k_used") is not None and util.get("bram_36k_available"):
        if util["bram_36k_used"] > util["bram_36k_available"]:
            point["status"] = "failed_bram_overutilization"
    if util.get("lut_used") is not None and util.get("lut_available"):
        if util["lut_used"] > util["lut_available"]:
            point["status"] = "failed_lut_overutilization"
    return point


def load_existing() -> Dict[str, Any]:
    if not OUT.exists():
        return {}
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def merge_points(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyfn = lambda p: (int(p.get("PROG_DEPTH") or 0), int(p.get("BUFFER_SIZE") or 0))
    merged = {keyfn(p): p for p in existing}
    for p in new:
        merged[keyfn(p)] = p
    return [merged[k] for k in sorted(merged.keys())]


def summarize(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    prog_pts = [p for p in points if int(p.get("BUFFER_SIZE") or 0) == 4096]
    buf_pts = [p for p in points if int(p.get("PROG_DEPTH") or 0) == 8192]
    closed_prog = [p for p in prog_pts if p.get("status") == "closed"]
    largest_closing = max((int(p["PROG_DEPTH"]) for p in closed_prog), default=None)
    p65 = next((p for p in prog_pts if int(p.get("PROG_DEPTH") or 0) == 65536), None)
    return {
        "largest_closing_PROG_DEPTH_at_buffer_4096": largest_closing,
        "prog_depth_65536_closes": bool(p65 and p65.get("status") == "closed"),
        "prog_depth_65536_status": None if p65 is None else p65.get("status"),
        "prog_depth_65536_point": p65,
        "prog_depth_points": prog_pts,
        "buffer_size_points_at_prog_8192": buf_pts,
        "trustworthy_capacity_note": (
            "Only status=closed with sane instr-BRAM scaling counts as capacity proof. "
            "failed_instr_bram_undercount rejects timing-only closes with collapsed RAMB36."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--skip-vivado", action="store_true", help="alias for --collect-only")
    ap.add_argument("--only-prog-depth", type=int, default=None)
    ap.add_argument("--only-buffer-size", type=int, default=None)
    ap.add_argument("--prog-depths", default=",".join(str(x) for x in PROG_DEPTHS))
    ap.add_argument("--buffer-sizes", default=",".join(str(x) for x in BUFFER_SIZES))
    args = ap.parse_args()
    collect_only = args.collect_only or args.skip_vivado

    prog_list = [int(x) for x in args.prog_depths.split(",") if x.strip()]
    raw_bufs = args.buffer_sizes.strip()
    if raw_bufs in ("", "-", "none", "NONE"):
        buf_list = []
    else:
        buf_list = [int(x) for x in raw_bufs.split(",") if x.strip()]
    if args.only_prog_depth:
        prog_list = [args.only_prog_depth]
        buf_list = []
    if args.only_buffer_size:
        buf_list = [args.only_buffer_size]
        if not args.only_prog_depth:
            prog_list = []

    new_points: List[Dict[str, Any]] = []

    # Analytical banking for all buffer sizes (even before synth).
    banking_table = [banking_check(b, ARRAY_SIZE, COMPUTE_DATA_WIDTH) for b in BUFFER_SIZES]

    jobs: List[Tuple[int, int, str]] = []
    for pd in prog_list:
        jobs.append((pd, 4096, f"prog_depth_pd{pd}_buf4096_mb{MAX_BATCH}_clk{int(CLOCK_PERIOD_NS)}_pd{PIPE_DEPTH}"))
    for bs in buf_list:
        # BUFFER sweep holds PROG_DEPTH at shipping 8192 unless banking fails.
        jobs.append((8192, bs, f"prog_depth_pd8192_buf{bs}_mb{MAX_BATCH}_clk{int(CLOCK_PERIOD_NS)}_pd{PIPE_DEPTH}"))

    # Dedupe (8192,4096) appears in both sweeps.
    seen = set()
    uniq_jobs = []
    for j in jobs:
        key = (j[0], j[1])
        if key in seen:
            continue
        seen.add(key)
        uniq_jobs.append(j)

    for prog_depth, buffer_size, prefix in uniq_jobs:
        bank = banking_check(buffer_size, ARRAY_SIZE, COMPUTE_DATA_WIDTH)
        if not bank["ok"]:
            new_points.append(
                {
                    "report_prefix": prefix,
                    "PROG_DEPTH": prog_depth,
                    "BUFFER_SIZE": buffer_size,
                    "status": "skipped_banking_nonintegral",
                    "banking": bank,
                    "clock_period_ns": CLOCK_PERIOD_NS,
                }
            )
            continue
        if not collect_only:
            if not VIVADO.exists():
                print("Vivado missing:", VIVADO, flush=True)
                return 2
            rc = run_vivado(prog_depth=prog_depth, buffer_size=buffer_size, report_prefix=prefix)
            print(f"vivado_rc={rc} prefix={prefix}", flush=True)
        new_points.append(collect_point(prefix, prog_depth=prog_depth, buffer_size=buffer_size))

    existing = load_existing()
    points = merge_points(list(existing.get("points") or []), new_points)
    summary = summarize(points)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "part": "xc7a100tcsg324-1",
        "methodology": {
            "flow": "program_arty_a7_revE.tcl full synth/impl",
            "shipping_datapath": {
                "COMPUTE_DATA_WIDTH": COMPUTE_DATA_WIDTH,
                "ACCUMULATOR_DATA_WIDTH": ACCUMULATOR_DATA_WIDTH,
                "ARRAY_SIZE": ARRAY_SIZE,
                "EXT_ADDR_EN": EXT_ADDR_EN,
                "MAX_BATCH_COUNT": MAX_BATCH,
                "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
                "QUANTIZER_LANES": ARRAY_SIZE,
            },
            "clock_period_ns": CLOCK_PERIOD_NS,
            "clock_period_rationale": (
                "Best closing period for mb48/pd3 in timing_closure_sweep.json (WNS=+2.949 @ 20 ns)"
            ),
            "baseline_bram_free_note": (
                "baseline_8x8_current_rtl_synth.json closed_config used 21/135 BRAM36; "
                "~114 free on Artix A7-100T"
            ),
        },
        "banking_integrity_n8_int8": banking_table,
        "points": points,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
