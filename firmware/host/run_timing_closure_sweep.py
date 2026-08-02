#!/usr/bin/env python3
"""Timing-closure sweep for 8x8 current-RTL (pipelined requant variant).

Writes bench/results/timing_closure_sweep.json from Vivado report prefixes
requant_fmax_mb{MB}_clk{PERIOD}_pd{DEPTH}_*.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "build" / "reports"
OUT = REPO / "bench" / "results" / "timing_closure_sweep.json"
VIVADO = Path(r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def dsp_from_util(text: str) -> Optional[int]:
    for line in text.splitlines():
        if "DSP48E1 only" in line:
            nums = [int(p.strip()) for p in line.split("|") if re.fullmatch(r"\s*\d+\s*", p)]
            if nums:
                return nums[0]
    m = re.search(r"\|\s*DSPs\s*\|\s*(\d+)\s*\|", text)
    return int(m.group(1)) if m else None


def parse_util(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {}
    for label, key in [
        ("Slice LUTs", "lut_used"),
        ("Slice Registers", "ff_used"),
        ("Block RAM Tile", "bram_36k_used"),
    ]:
        m = re.search(
            rf"\|\s*{re.escape(label)}[^|]*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            text,
        )
        if m:
            out[key] = int(m.group(1))
            out[key.replace("_used", "_available")] = int(m.group(2))
    dsp = dsp_from_util(text)
    if dsp is not None:
        out["dsp_used"] = dsp
        out["dsp_available"] = 240
    return out


def parse_timing(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {"raw_path": str(path.relative_to(REPO)).replace("\\", "/")}
    m = re.search(r"Worst Slack\s+([-\d\.]+)ns,\s+Total Violation\s+([-\d\.]+)ns", text)
    if m:
        out["wns_ns"] = float(m.group(1))
        out["tns_ns"] = float(m.group(2))
    m = re.search(r"Hold\s*:\s*\d+\s+Failing Endpoints,\s+Worst Slack\s+([-\d\.]+)ns", text)
    if m:
        out["whs_ns"] = float(m.group(1))
    m = re.search(r"Data Path Delay:\s+([-\d\.]+)ns", text)
    if m:
        out["data_path_delay_ns"] = float(m.group(1))
    m = re.search(r"Source:\s+(\S+)", text)
    if m:
        out["critical_source"] = m.group(1)
    m = re.search(r"Destination:\s+(\S+)", text)
    if m:
        out["critical_destination"] = m.group(1)
    # DSP A->P vs registered: Prop_dsp48e1_A*_P* means combo through DSP.
    if "Prop_dsp48e1_A" in text and "_P[" in text:
        out["dsp_path_style"] = "A_to_P_combo_no_MREG"
    if "Prop_dsp48e1_CLK_P" in text or "MREG" in text:
        out["dsp_path_style_hint"] = "may_include_registered_dsp"
    out["all_paths_met"] = bool(out.get("wns_ns") is not None and out["wns_ns"] >= 0.0)
    return out


def run_vivado(
    *,
    clock_period_ns: float,
    max_batch: int,
    pipe_depth: int,
    report_prefix: str,
) -> int:
    log_dir = REPO / "build" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    proj = f"build/vivado_arty_a7_fmax_mb{max_batch}_clk{clock_period_ns:g}_pd{pipe_depth}"
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
        f"uTPU_fmax_mb{max_batch}_pd{pipe_depth}",
        "PROG_DEPTH",
        "8192",
        "COMPUTE_DATA_WIDTH",
        "8",
        "ACCUMULATOR_DATA_WIDTH",
        "32",
        "ARRAY_SIZE",
        "8",
        "BUFFER_SIZE",
        "4096",
        "EXT_ADDR_EN",
        "1",
        "MAX_BATCH_COUNT",
        str(max_batch),
        "QUANTIZER_LANES",
        "8",
        "RELU_LANES",
        "8",
        "QUANTIZER_PIPE_DEPTH",
        str(pipe_depth),
        "clock_period",
        str(clock_period_ns),
        "report_prefix",
        report_prefix,
        "do_program",
        "0",
    ]
    print("RUN", " ".join(args[5:20]), "...")
    return subprocess.call(args, cwd=str(REPO))


def collect_point(prefix: str, clock_period_ns: float, max_batch: int, pipe_depth: int) -> Dict[str, Any]:
    util_p = REPORTS / f"{prefix}_utilization.rpt"
    tim_p = REPORTS / f"{prefix}_timing_summary.rpt"
    route_p = REPORTS / f"{prefix}_route_status.rpt"
    point: Dict[str, Any] = {
        "report_prefix": prefix,
        "clock_period_ns": clock_period_ns,
        "frequency_mhz_constraint": 1000.0 / clock_period_ns,
        "MAX_BATCH_COUNT": max_batch,
        "QUANTIZER_PIPE_DEPTH": pipe_depth,
        "status": "missing_reports",
    }
    if not util_p.exists() or not tim_p.exists():
        if route_p.exists():
            point["route_status"] = route_p.read_text(encoding="utf-8", errors="replace")[:500]
        return point
    util = parse_util(util_p)
    tim = parse_timing(tim_p)
    point["utilization"] = util
    point["timing"] = tim
    wns = tim.get("wns_ns")
    if wns is not None and wns >= 0:
        # Achieved Fmax estimate from measured data path (+skew via period-WNS).
        point["status"] = "closed"
        point["achieved_fmax_mhz_from_period_minus_wns"] = 1000.0 / (clock_period_ns - wns)
        if tim.get("data_path_delay_ns") is not None:
            point["achieved_fmax_mhz_from_data_path_delay"] = 1000.0 / tim["data_path_delay_ns"]
    else:
        point["status"] = "failed_timing" if wns is not None else "unknown"
    return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe-depth", type=int, default=3)
    parser.add_argument(
        "--periods",
        type=float,
        nargs="+",
        default=[20.0, 18.0, 17.0, 16.0, 15.0],
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--run", action="store_true", help="Launch Vivado for missing points")
    parser.add_argument("--include-aggressive", action="store_true", help="Also try 12 and 10 ns")
    args = parser.parse_args()

    periods = list(args.periods)
    if args.include_aggressive:
        periods.extend([12.0, 10.0])

    points: List[Dict[str, Any]] = []
    for mb in args.batches:
        for period in periods:
            prefix = f"requant_fmax_mb{mb}_clk{period:g}_pd{args.pipe_depth}"
            util_p = REPORTS / f"{prefix}_utilization.rpt"
            if args.run and not util_p.exists():
                rc = run_vivado(
                    clock_period_ns=period,
                    max_batch=mb,
                    pipe_depth=args.pipe_depth,
                    report_prefix=prefix,
                )
                if rc != 0:
                    print(f"WARN vivado rc={rc} for {prefix}")
            points.append(collect_point(prefix, period, mb, args.pipe_depth))

            # Skip tighter clocks for this MB if 15 ns already failed.
            if period == 15.0 and points[-1].get("status") == "failed_timing":
                if not args.include_aggressive:
                    break

    closed = [p for p in points if p.get("status") == "closed"]
    best = None
    if closed:
        best = max(closed, key=lambda p: p["frequency_mhz_constraint"])

    artifact = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "part": "xc7a100tcsg324-1",
        "methodology": (
            "Full top synth/impl via program_arty_a7_revE.tcl. ARRAY_SIZE=8 INT8, "
            "QUANTIZER_LANES=8, QUANTIZER_PIPE_DEPTH selectable. Grid: clock_period_ns x "
            "MAX_BATCH_COUNT. Skip 12/10 unless --include-aggressive or 15 closes."
        ),
        "headroom_note": (
            "At a closed point with constraint T and WNS W, the effective critical path "
            "is approximately T-W (not 'headroom toward 100 MHz'). Vivado may improve "
            "paths when re-constrained tighter (expect ~15-20%, not 2x)."
        ),
        "points": points,
        "highest_closing": best,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} points={len(points)} closed={len(closed)}")
    if best:
        print(
            "best",
            best["report_prefix"],
            "constraint_mhz",
            best["frequency_mhz_constraint"],
            "wns",
            best["timing"].get("wns_ns"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
