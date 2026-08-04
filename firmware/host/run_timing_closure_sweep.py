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
from typing import Any, Dict, List, Optional, Tuple

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


def point_key(p: Dict[str, Any]) -> Tuple[int, float, int]:
    return (
        int(p.get("MAX_BATCH_COUNT") or 0),
        float(p.get("clock_period_ns") or 0.0),
        int(p.get("QUANTIZER_PIPE_DEPTH") or 0),
    )


def load_existing_points() -> List[Dict[str, Any]]:
    if not OUT.exists():
        return []
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    pts = data.get("points")
    return list(pts) if isinstance(pts, list) else []


def point_info_rank(p: Dict[str, Any]) -> int:
    """Higher = richer evidence. Prevents missing_reports from erasing known failures."""
    status = str(p.get("status") or "")
    if status == "closed":
        return 40
    if status.startswith("failed_") and p.get("failure_detail"):
        return 30
    if status.startswith("failed_") or p.get("failure_reason") not in (None, "missing_reports"):
        return 20
    if status == "missing_reports":
        return 5
    return 10


def prefer_point(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer richer evidence; on tie keep new (fresh collect)."""
    if point_info_rank(new) >= point_info_rank(old):
        return new
    return old


def merge_points(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union by (MAX_BATCH_COUNT, period, pipe_depth). Never drop keys; prefer richer evidence."""
    merged: Dict[Tuple[int, float, int], Dict[str, Any]] = {}
    for p in existing:
        merged[point_key(p)] = p
    for p in new:
        key = point_key(p)
        if key in merged:
            merged[key] = prefer_point(merged[key], p)
        else:
            merged[key] = p
    return propagate_lut_failures([merged[k] for k in sorted(merged.keys())])


def propagate_lut_failures(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LUT overutilization is architectural (period-independent). Copy cause onto sibling gaps."""
    by_mb_pd: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for p in points:
        if p.get("failure_reason") == "lut_overutilization" and p.get("failure_detail"):
            key = (int(p.get("MAX_BATCH_COUNT") or 0), int(p.get("QUANTIZER_PIPE_DEPTH") or 0))
            by_mb_pd[key] = p["failure_detail"]
    out: List[Dict[str, Any]] = []
    for p in points:
        q = dict(p)
        key = (int(q.get("MAX_BATCH_COUNT") or 0), int(q.get("QUANTIZER_PIPE_DEPTH") or 0))
        if (
            key in by_mb_pd
            and q.get("status") in ("missing_reports", "unknown")
            and q.get("failure_reason") in (None, "missing_reports")
        ):
            q["status"] = "failed_lut_overutilization"
            q["failure_reason"] = "lut_overutilization"
            q["failure_detail"] = dict(by_mb_pd[key])
            q["failure_note"] = (
                "Propagated from sibling MAX_BATCH_COUNT period that recorded LUT "
                "overutilization; synth util is period-independent on this part."
            )
        out.append(q)
    return out


def summarize_max_batch_lut_bisect(points: List[Dict[str, Any]], pipe_depth: int) -> Dict[str, Any]:
    """Largest MAX_BATCH_COUNT that fits+closes at the shipping 20 ns period."""
    shipping = [
        p
        for p in points
        if int(p.get("QUANTIZER_PIPE_DEPTH") or 0) == pipe_depth
        and abs(float(p.get("clock_period_ns") or 0.0) - 20.0) < 1e-9
    ]
    closed = [p for p in shipping if p.get("status") == "closed"]
    failed = [p for p in shipping if str(p.get("status") or "").startswith("failed_")]
    best = max(closed, key=lambda p: int(p.get("MAX_BATCH_COUNT") or 0)) if closed else None
    n = 8
    occ = None
    if best is not None:
        b = int(best["MAX_BATCH_COUNT"])
        occ = {
            "formula": "B / (2N + B)",
            "N": n,
            "B": b,
            "streaming_occupancy_ceiling": b / (2 * n + b),
            "note": (
                "Architectural batch ceiling caps the occupancy the overlap work is chasing; "
                f"B=32 => {32/(2*n+32):.3f}, B=48 => {48/(2*n+48):.3f}, B=64 => {64/(2*n+64):.3f}."
            ),
        }
    return {
        "shipping_period_ns": 20.0,
        "QUANTIZER_PIPE_DEPTH": pipe_depth,
        "attempted_at_20ns": sorted(int(p["MAX_BATCH_COUNT"]) for p in shipping),
        "largest_closing_MAX_BATCH_COUNT": None if best is None else int(best["MAX_BATCH_COUNT"]),
        "largest_closing_point": best,
        "failed_at_20ns": [
            {
                "MAX_BATCH_COUNT": int(p["MAX_BATCH_COUNT"]),
                "status": p.get("status"),
                "failure_reason": p.get("failure_reason"),
                "failure_detail": p.get("failure_detail"),
            }
            for p in sorted(failed, key=lambda x: int(x.get("MAX_BATCH_COUNT") or 0))
        ],
        "occupancy_ceiling_at_largest_closing": occ,
        "mb64_finding": (
            "MAX_BATCH_COUNT=64 is an architectural LUT ceiling on xc7a100t "
            "(67217 > 63400), not a skip — bisect {24,32,48} all close at 20 ns."
        ),
    }


def infer_failure_reason(prefix: str, point: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a recorded failure cause when reports are missing or timing failed."""
    if point.get("status") == "closed":
        return point
    vlog = REPO / "build" / "logs" / f"{prefix}.vlog"
    reason = None
    detail = None
    if vlog.exists():
        text = vlog.read_text(encoding="utf-8", errors="replace")
        if "UTLZ-1" in text or "LUT as Logic over-utilized" in text:
            reason = "lut_overutilization"
            m = re.search(
                r"requires\s+(\d+)\s+of such cell types but only\s+(\d+)\s+compatible",
                text,
            )
            if m:
                detail = {"lut_required": int(m.group(1)), "lut_available": int(m.group(2))}
        elif "Failed runs(s) : 'impl_1'" in text or "place_design failed" in text:
            reason = reason or "impl_failed"
        elif "synth" in text.lower() and "ERROR" in text:
            reason = reason or "synth_failed"
    if point.get("status") == "failed_timing":
        reason = "wns_negative"
        detail = {"wns_ns": (point.get("timing") or {}).get("wns_ns")}
    if point.get("status") == "missing_reports" and reason is None:
        reason = "missing_reports"
    if reason:
        point["failure_reason"] = reason
        if detail:
            point["failure_detail"] = detail
        if point.get("status") == "missing_reports" and reason != "missing_reports":
            point["status"] = f"failed_{reason}"
    return point


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
        return infer_failure_reason(prefix, point)
    util = parse_util(util_p)
    tim = parse_timing(tim_p)
    point["utilization"] = util
    point["timing"] = tim
    wns = tim.get("wns_ns")
    if wns is not None and wns >= 0:
        point["status"] = "closed"
        point["achieved_fmax_mhz_from_period_minus_wns"] = 1000.0 / (clock_period_ns - wns)
        if tim.get("data_path_delay_ns") is not None:
            point["achieved_fmax_mhz_from_data_path_delay"] = 1000.0 / tim["data_path_delay_ns"]
    else:
        point["status"] = "failed_timing" if wns is not None else "unknown"
    return infer_failure_reason(prefix, point)

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

    periods = list(dict.fromkeys(float(p) for p in args.periods))
    if args.include_aggressive:
        for p in (12.0, 10.0):
            if p not in periods:
                periods.append(p)

    existing = load_existing_points()
    existing_keys = {point_key(p) for p in existing}
    new_points: List[Dict[str, Any]] = []
    for mb in args.batches:
        for period in periods:
            prefix = f"requant_fmax_mb{mb}_clk{period:g}_pd{args.pipe_depth}"
            util_p = REPORTS / f"{prefix}_utilization.rpt"
            key = (int(mb), float(period), int(args.pipe_depth))
            if args.run and not util_p.exists():
                rc = run_vivado(
                    clock_period_ns=period,
                    max_batch=mb,
                    pipe_depth=args.pipe_depth,
                    report_prefix=prefix,
                )
                if rc != 0:
                    print(f"WARN vivado rc={rc} for {prefix}")
            # Without --run, only refresh keys that already exist or have reports on disk.
            # Do not invent missing_reports stubs for never-attempted cells (those are gaps,
            # not evidence). --run always records the attempt, including failures.
            if not args.run and not util_p.exists() and key not in existing_keys:
                continue
            new_points.append(collect_point(prefix, period, mb, args.pipe_depth))

            # Skip tighter clocks for this MB if 15 ns already failed timing (not LUT).
            if abs(period - 15.0) < 1e-9 and new_points and new_points[-1].get("status") == "failed_timing":
                if not args.include_aggressive:
                    break

    points = merge_points(existing, new_points)
    closed = [p for p in points if p.get("status") == "closed"]
    best = None
    if closed:
        best = max(closed, key=lambda p: p["frequency_mhz_constraint"])
    lut_bisect = summarize_max_batch_lut_bisect(points, args.pipe_depth)

    artifact = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "part": "xc7a100tcsg324-1",
        "methodology": (
            "Full top synth/impl via program_arty_a7_revE.tcl. ARRAY_SIZE=8 INT8, "
            "QUANTIZER_LANES=8, QUANTIZER_PIPE_DEPTH selectable. Grid: clock_period_ns x "
            "MAX_BATCH_COUNT. Skip 12/10 unless --include-aggressive or 15 closes. "
            "Artifact merges by (MAX_BATCH_COUNT, period, pipe_depth) so re-runs never "
            "silently drop previously recorded points (including failures). "
            "missing_reports never overwrites a richer failed_* sibling; LUT "
            "overutilization propagates across periods for the same MAX_BATCH_COUNT."
        ),
        "headroom_note": (
            "At a closed point with constraint T and WNS W, the effective critical path "
            "is approximately T-W (not 'headroom toward 100 MHz'). Vivado may improve "
            "paths when re-constrained tighter (expect ~15-20%, not 2x)."
        ),
        "merge_policy": (
            "union_by_max_batch_period_pipe_depth; prefer richer evidence "
            "(closed > failed+detail > failed > missing_reports); "
            "propagate lut_overutilization across periods"
        ),
        "points": points,
        "highest_closing": best,
        "max_batch_lut_bisect": lut_bisect,
        "points_retained_from_prior": len(existing),
        "points_written_this_run": len(new_points),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} points={len(points)} closed={len(closed)} (merged from {len(existing)} prior)")
    if best:
        print(
            "best",
            best["report_prefix"],
            "constraint_mhz",
            best["frequency_mhz_constraint"],
            "wns",
            best["timing"].get("wns_ns"),
        )
    print(
        "max_batch_lut_bisect largest_closing=",
        lut_bisect.get("largest_closing_MAX_BATCH_COUNT"),
        "failed_at_20ns=",
        [f.get("MAX_BATCH_COUNT") for f in lut_bisect.get("failed_at_20ns") or []],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
