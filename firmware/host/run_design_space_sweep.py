#!/usr/bin/env python3
"""Design-space / roofline sweep at PROG_DEPTH=65536 (Artix A7-100T).

Grid (default):
  ARRAY_SIZE {4,8} x COMPUTE_DATA_WIDTH {4,8} x MAX_BATCH_COUNT {4,16,48}
  x clock_period {20,15,12,10} ns
  + optional 16x16 INT4 Pareto point
Fixed: PROG_DEPTH=65536, QUANTIZER_PIPE_DEPTH=3, BUFFER_SIZE=4096, EXT_ADDR_EN=1.

Reuses closed reports under build/reports/ when parameters match (including the
shipping prog_depth_pd65536_buf4096_mb48_clk20_pd3 alias). Only launches Vivado
for missing points. Priority order: shipping-relevant N=8 INT8 mb{4,16,48}
periods first, then expand.

Emits:
  bench/results/design_space_sweep.json
  docs/design_space_roofline.png
  docs/HARDWARE_DESIGN_SPACE.md

Peak/achieved GOP/s are computed ONCE per (ARRAY_SIZE, CDW, MAX_BATCH_COUNT)
from that config's tightest closing period — not once per period row.
Per-period rows remain timing evidence (period, WNS, util, margin_class).
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
HOST = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from run_timing_closure_sweep import (  # noqa: E402
    VIVADO,
    parse_timing,
    parse_util,
)

OUT = REPO / "bench" / "results" / "design_space_sweep.json"
REPORTS = REPO / "build" / "reports"
DOCS = REPO / "docs"
PLOT = DOCS / "design_space_roofline.png"
MD = DOCS / "HARDWARE_DESIGN_SPACE.md"
TCL = REPO / "scripts" / "synth_design_space_point.tcl"

PROG_DEPTH = 65536
PIPE_DEPTH = 3
BUFFER_SIZE = 4096
EXT_ADDR_EN = 1
PART = "xc7a100tcsg324-1"

ARRAY_SIZES = [4, 8]
CDWS = [4, 8]
BATCHES = [4, 16, 48]
PERIODS = [20.0, 15.0, 12.0, 10.0]

# Known aliases: (N, CDW, MB, period) -> existing report prefix with matching params.
REPORT_ALIASES: Dict[Tuple[int, int, int, float], str] = {
    # Shipping capacity close: N=8 INT8 mb48 @20ns, PROG_DEPTH=65536, pd3.
    (8, 8, 48, 20.0): "prog_depth_pd65536_buf4096_mb48_clk20_pd3",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def accum_width(cdw: int) -> int:
    return 32 if int(cdw) >= 8 else 16


def period_tag(period: float) -> str:
    f = float(period)
    if f.is_integer():
        return str(int(f))
    return str(f).replace(".", "p")


def report_prefix(n: int, cdw: int, mb: int, period: float) -> str:
    # Prefer canonical dss_ prefix; callers resolve aliases separately.
    return f"dss_n{n}_cdw{cdw}_mb{mb}_clk{period_tag(period)}_pd{PIPE_DEPTH}_prog{PROG_DEPTH}"


def point_key(
    n: int, cdw: int, mb: int, period: float
) -> Tuple[int, int, int, float, int, int]:
    return (int(n), int(cdw), int(mb), float(period), PROG_DEPTH, PIPE_DEPTH)


def point_key_from_dict(p: Dict[str, Any]) -> Tuple[int, int, int, float, int, int]:
    return (
        int(p.get("ARRAY_SIZE") or 0),
        int(p.get("COMPUTE_DATA_WIDTH") or 0),
        int(p.get("MAX_BATCH_COUNT") or 0),
        float(p.get("clock_period_ns") or 0.0),
        int(p.get("PROG_DEPTH") or PROG_DEPTH),
        int(p.get("QUANTIZER_PIPE_DEPTH") or PIPE_DEPTH),
    )


def resolve_prefix(n: int, cdw: int, mb: int, period: float) -> str:
    alias = REPORT_ALIASES.get((int(n), int(cdw), int(mb), float(period)))
    canonical = report_prefix(n, cdw, mb, period)
    if alias:
        util_a = REPORTS / f"{alias}_utilization.rpt"
        tim_a = REPORTS / f"{alias}_timing_summary.rpt"
        if util_a.exists() and tim_a.exists():
            return alias
    return canonical


def reports_present(prefix: str) -> bool:
    return (REPORTS / f"{prefix}_utilization.rpt").exists() and (
        REPORTS / f"{prefix}_timing_summary.rpt"
    ).exists()


def load_occupancy() -> Dict[str, Any]:
    steady = REPO / "bench" / "results" / "steady_state_attribution.json"
    if steady.exists():
        data = json.loads(steady.read_text(encoding="utf-8"))
        # Prefer explicit occupancy / compute share fields when present.
        occ = data.get("compute_occupancy")
        if occ is None:
            occ = data.get("occupancy")
        if occ is None:
            groups = data.get("groups") or {}
            compute = (groups.get("compute") or {}).get("cycles")
            total = data.get("total_program_cycles")
            if compute is not None and total:
                occ = float(compute) / float(total)
        if occ is None:
            share = data.get("compute_share")
            if share is not None:
                occ = float(share)
        if occ is None:
            raise ValueError("steady_state_attribution.json present but no occupancy field")
        ss = data.get("steady_state") or {}
        note = (
            ss.get("occupancy_source")
            or data.get("standing_rule_5")
            or "Occupancy from steady_state_attribution.json (may be cold proxy)."
        )
        return {
            "occupancy": float(occ),
            "occupancy_source": "steady_state_attribution.json",
            "occupancy_note": str(note),
            "artifact": str(steady.relative_to(REPO)).replace("\\", "/"),
        }

    mnist = REPO / "bench" / "results" / "cycle_attribution_mnist.json"
    data = json.loads(mnist.read_text(encoding="utf-8"))
    compute = int((data.get("groups") or {}).get("compute", {}).get("cycles") or 0)
    total = int(data.get("total_program_cycles") or 0)
    occ = (compute / total) if total else 0.0
    return {
        "occupancy": float(occ),
        "occupancy_source": (
            "cycle_attribution_mnist.json post-widen placeholder "
            f"(compute {compute}/{total}); steady_state_attribution.json absent"
        ),
        "occupancy_note": (
            "Placeholder cold-path compute share until buffer-resident / steady-state "
            "attribution lands. NOT peak; applied as achieved = peak * occupancy."
        ),
        "compute_cycles": compute,
        "total_program_cycles": total,
        "artifact": str(mnist.relative_to(REPO)).replace("\\", "/"),
    }


def load_accuracy_pair() -> Dict[str, Any]:
    path = REPO / "bench" / "results" / "real_model_accelerator.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    acc = data.get("accuracy_comparison") or {}
    int8 = float((acc.get("int8") or {}).get("per_layer_accuracy") or 0.9733)
    int4 = float((acc.get("int4") or {}).get("per_layer_accuracy") or 0.5832)
    return {
        "int8_per_layer_accuracy": int8,
        "int4_per_layer_accuracy": int4,
        "artifact": str(path.relative_to(REPO)).replace("\\", "/"),
        "note": (
            "Pareto uses measured per-layer accuracies 97.33% (INT8) vs 58.32% (INT4) "
            "from real_model_accelerator.json; not per-channel INT4."
        ),
    }


def binding_from_util(util: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the resource that is oversubscribed, if any."""
    checks = [
        ("lut", "lut_used", "lut_available"),
        ("dsp", "dsp_used", "dsp_available"),
        ("bram", "bram_36k_used", "bram_36k_available"),
        ("ff", "ff_used", "ff_available"),
    ]
    binders = []
    for name, u_key, a_key in checks:
        used = util.get(u_key)
        avail = util.get(a_key)
        if used is None or avail is None or avail <= 0:
            continue
        frac = float(used) / float(avail)
        binders.append((frac, name, int(used), int(avail)))
    if not binders:
        return None
    binders.sort(reverse=True)
    frac, name, used, avail = binders[0]
    if used > avail:
        return {
            "binding_resource": name,
            "used": used,
            "available": avail,
            "oversubscribed": True,
            "utilization_fraction": frac,
        }
    # Near-full (>95%) is the soft binder even when timing also fails.
    if frac >= 0.95:
        return {
            "binding_resource": name,
            "used": used,
            "available": avail,
            "oversubscribed": False,
            "near_full": True,
            "utilization_fraction": frac,
        }
    return {
        "binding_resource": name,
        "used": used,
        "available": avail,
        "oversubscribed": False,
        "utilization_fraction": frac,
        "note": "Highest util resource (not necessarily oversubscribed).",
    }


def infer_failure(prefix: str, point: Dict[str, Any]) -> Dict[str, Any]:
    if point.get("status") == "closed":
        return point
    vlog = REPO / "build" / "logs" / f"{prefix}.vlog"
    reason = point.get("failure_reason")
    detail = point.get("failure_detail")
    binding = point.get("binding_resource")

    util = point.get("utilization") or {}
    bind_info = binding_from_util(util) if util else None

    if vlog.exists():
        text = vlog.read_text(encoding="utf-8", errors="replace")
        if "UTLZ-1" in text or "LUT as Logic over-utilized" in text:
            reason = "lut_overutilization"
            binding = "lut"
            m = re.search(
                r"requires\s+(\d+)\s+of such cell types but only\s+(\d+)\s+compatible",
                text,
            )
            if m:
                detail = {"lut_required": int(m.group(1)), "lut_available": int(m.group(2))}
        elif re.search(r"DSP.*over-utilized|requires\s+\d+\s+.*DSP", text, re.I):
            reason = "dsp_overutilization"
            binding = "dsp"
        elif re.search(r"Block RAM.*over-utilized|RAMB.*over-utilized", text, re.I):
            reason = "bram_overutilization"
            binding = "bram"
        elif "Synthesis failed" in text or re.search(r"ERROR: \[Synth", text):
            reason = reason or "synth_failed"
        elif "Failed runs(s) : 'impl_1'" in text or "place_design failed" in text:
            reason = reason or "impl_failed"

    if point.get("status") == "failed_timing":
        reason = "wns_negative"
        detail = {"wns_ns": (point.get("timing") or {}).get("wns_ns")}
        if bind_info and bind_info.get("near_full"):
            binding = bind_info["binding_resource"]
            point["binding_detail"] = bind_info
        else:
            binding = binding or "timing"

    if bind_info and bind_info.get("oversubscribed"):
        binding = bind_info["binding_resource"]
        reason = reason or f"{binding}_overutilization"
        point["binding_detail"] = bind_info
    elif bind_info and not binding:
        point["binding_detail"] = bind_info
        binding = bind_info.get("binding_resource")

    if point.get("status") == "missing_reports" and reason is None:
        reason = "missing_reports"

    if reason:
        point["failure_reason"] = reason
        if detail:
            point["failure_detail"] = detail
        if point.get("status") == "missing_reports" and reason != "missing_reports":
            point["status"] = f"failed_{reason}"
    if binding:
        point["binding_resource"] = binding
    return point


FMAX_DERIVATION = {
    "formula": "achieved_fmax_mhz_from_period_minus_wns = 1000 / (clock_period_ns - WNS_ns)",
    "note": (
        "This understates capability at loose constraints: Vivado stops optimizing "
        "once the applied period is met, so loose-period (period−WNS) Fmax values "
        "are floors, not measurements of the design's true speed. Throughput "
        "(peak/achieved GOP/s) is therefore taken from each configuration's "
        "tightest closing period only."
    ),
}


def margin_class_for_wns(wns: Optional[float]) -> Optional[str]:
    if wns is None:
        return None
    w = float(wns)
    if w > 1.0:
        return "comfortable"  # >1 ns
    if w >= 0.2:
        return "thin"  # 0.2–1 ns
    return "marginal"  # <0.2 ns (includes 12 ps closes)


def attach_throughput(point: Dict[str, Any], occupancy: float) -> Dict[str, Any]:
    """Annotate timing evidence on a period row. Do NOT attach roofline GOP/s here."""
    status = point.get("status")
    period = float(point.get("clock_period_ns") or 0.0)
    wns = (point.get("timing") or {}).get("wns_ns")
    if status == "closed":
        if point.get("achieved_fmax_mhz_from_period_minus_wns") is None:
            if period and wns is not None and (period - float(wns)) > 0:
                point["achieved_fmax_mhz_from_period_minus_wns"] = 1000.0 / (
                    period - float(wns)
                )
        point["margin_class"] = margin_class_for_wns(
            None if wns is None else float(wns)
        )
        point["fmax_derivation"] = dict(FMAX_DERIVATION)
    # Period rows are timing evidence only — throughput lives in throughput_by_config.
    point["peak_gops"] = None
    point["achieved_gops"] = None
    point["occupancy_applied"] = float(occupancy)
    point["throughput_note"] = (
        "Per-period rows are timing evidence only; plotted GOP/s come from "
        "throughput_by_config (one entry per ARRAY_SIZE×CDW×MAX_BATCH_COUNT, "
        "tightest closing period)."
    )
    return point


def config_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(p.get("ARRAY_SIZE") or 0),
        int(p.get("COMPUTE_DATA_WIDTH") or 0),
        int(p.get("MAX_BATCH_COUNT") or 0),
    )


def build_throughput_by_config(
    points: List[Dict[str, Any]], occupancy: float
) -> List[Dict[str, Any]]:
    """One throughput record per hardware config from its tightest closing period."""
    best: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for p in points:
        if p.get("status") != "closed":
            continue
        key = config_key(p)
        period = float(p.get("clock_period_ns") or 0.0)
        prev = best.get(key)
        if prev is None or period < float(prev.get("clock_period_ns") or 1e9):
            best[key] = p
    rows: List[Dict[str, Any]] = []
    for key in sorted(best.keys()):
        p = best[key]
        n, cdw, mb = key
        fmax = p.get("achieved_fmax_mhz_from_period_minus_wns")
        wns = (p.get("timing") or {}).get("wns_ns")
        period = float(p.get("clock_period_ns") or 0.0)
        if fmax is None and period and wns is not None and (period - float(wns)) > 0:
            fmax = 1000.0 / (period - float(wns))
        peak = None
        achieved = None
        if fmax is not None and n > 0:
            peak = (n * n) * 2.0 * (float(fmax) / 1000.0)
            achieved = peak * float(occupancy)
        rows.append(
            {
                "ARRAY_SIZE": n,
                "COMPUTE_DATA_WIDTH": cdw,
                "MAX_BATCH_COUNT": mb,
                "source_clock_period_ns": period,
                "source_frequency_mhz_constraint": 1000.0 / period if period else None,
                "source_report_prefix": p.get("report_prefix"),
                "source_wns_ns": wns,
                "margin_class": margin_class_for_wns(
                    None if wns is None else float(wns)
                ),
                "achieved_fmax_mhz_from_period_minus_wns": fmax,
                "fmax_derivation": dict(FMAX_DERIVATION),
                "peak_gops": peak,
                "achieved_gops": achieved,
                "occupancy_applied": float(occupancy),
                "throughput_formulas": {
                    "peak_gops": "ARRAY_SIZE^2 * 2 * Fmax_GHz",
                    "achieved_gops": "peak_gops * occupancy",
                    "selection": "tightest closing clock_period_ns for this config",
                    "note": "peak and achieved are separate; never report peak as achieved",
                },
            }
        )
    return rows


def point_was_attempted(p: Dict[str, Any]) -> bool:
    """True if this cell has real evidence (not a never-tried placeholder)."""
    st = str(p.get("status") or "")
    if st == "closed":
        return True
    if st.startswith("failed_"):
        return True
    if st == "skipped":
        return True
    # missing_reports / unknown = not attempted (or incomplete harvest)
    return False


def grid_cell_dict(n: int, cdw: int, mb: int, period: float) -> Dict[str, Any]:
    return {
        "ARRAY_SIZE": int(n),
        "COMPUTE_DATA_WIDTH": int(cdw),
        "MAX_BATCH_COUNT": int(mb),
        "clock_period_ns": float(period),
        "canonical_prefix": report_prefix(n, cdw, mb, period),
    }


def collect_point(
    *,
    n: int,
    cdw: int,
    mb: int,
    period: float,
    occupancy: float,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    prefix = prefix or resolve_prefix(n, cdw, mb, period)
    util_p = REPORTS / f"{prefix}_utilization.rpt"
    tim_p = REPORTS / f"{prefix}_timing_summary.rpt"
    # Synth-only fallback report name used when impl never ran.
    if not util_p.exists():
        synth_util = REPORTS / f"{prefix}_utilization_synth.rpt"
        if synth_util.exists():
            util_p = synth_util

    point: Dict[str, Any] = {
        "report_prefix": prefix,
        "ARRAY_SIZE": int(n),
        "COMPUTE_DATA_WIDTH": int(cdw),
        "ACCUMULATOR_DATA_WIDTH": accum_width(cdw),
        "MAX_BATCH_COUNT": int(mb),
        "clock_period_ns": float(period),
        "frequency_mhz_constraint": 1000.0 / float(period),
        "PROG_DEPTH": PROG_DEPTH,
        "BUFFER_SIZE": BUFFER_SIZE,
        "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
        "QUANTIZER_LANES": int(n),
        "RELU_LANES": int(n),
        "EXT_ADDR_EN": EXT_ADDR_EN,
        "status": "missing_reports",
    }

    if not util_p.exists() or not tim_p.exists():
        # Util-only (synth fail after util dump).
        if util_p.exists() and not tim_p.exists():
            util = parse_util(util_p)
            point["utilization"] = util
            point["status"] = "failed_synth_or_impl"
            bind = binding_from_util(util)
            if bind:
                point["binding_detail"] = bind
                if bind.get("oversubscribed"):
                    point["binding_resource"] = bind["binding_resource"]
                    point["status"] = f"failed_{bind['binding_resource']}_overutilization"
                    point["failure_reason"] = f"{bind['binding_resource']}_overutilization"
            return attach_throughput(infer_failure(prefix, point), occupancy)
        return attach_throughput(infer_failure(prefix, point), occupancy)

    util = parse_util(util_p)
    tim = parse_timing(tim_p)
    point["utilization"] = util
    point["timing"] = tim
    wns = tim.get("wns_ns")
    if wns is not None and wns >= 0:
        point["status"] = "closed"
        point["achieved_fmax_mhz_from_period_minus_wns"] = 1000.0 / (float(period) - float(wns))
        if tim.get("data_path_delay_ns"):
            point["achieved_fmax_mhz_from_data_path_delay"] = 1000.0 / float(
                tim["data_path_delay_ns"]
            )
        bind = binding_from_util(util)
        if bind:
            point["binding_detail"] = bind
            point["binding_resource"] = bind.get("binding_resource")
    else:
        point["status"] = "failed_timing" if wns is not None else "unknown"
    return attach_throughput(infer_failure(prefix, point), occupancy)


def run_vivado(
    *,
    n: int,
    cdw: int,
    mb: int,
    period: float,
    prefix: str,
    jobs: int = 4,
) -> int:
    log_dir = REPO / "build" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    args = [
        str(VIVADO),
        "-mode",
        "batch",
        "-source",
        str(TCL.relative_to(REPO)).replace("\\", "/"),
        "-log",
        str(log_dir / f"{prefix}.vlog"),
        "-journal",
        str(log_dir / f"{prefix}.jou"),
        "-tclargs",
        "ARRAY_SIZE",
        str(n),
        "COMPUTE_DATA_WIDTH",
        str(cdw),
        "ACCUMULATOR_DATA_WIDTH",
        str(accum_width(cdw)),
        "MAX_BATCH_COUNT",
        str(mb),
        "PROG_DEPTH",
        str(PROG_DEPTH),
        "BUFFER_SIZE",
        str(BUFFER_SIZE),
        "EXT_ADDR_EN",
        str(EXT_ADDR_EN),
        "QUANTIZER_LANES",
        str(n),
        "RELU_LANES",
        str(n),
        "QUANTIZER_PIPE_DEPTH",
        str(PIPE_DEPTH),
        "clock_period",
        period_tag(period),
        "report_prefix",
        prefix,
        "jobs",
        str(jobs),
    ]
    print("RUN", prefix, flush=True)
    # Do not redirect stdout onto the same path as vivado -log (Windows file lock).
    tee_path = log_dir / f"{prefix}.stdout.log"
    with open(tee_path, "w", encoding="utf-8", errors="replace") as logf:
        return subprocess.call(args, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)


def priority_rank(n: int, cdw: int, mb: int, period: float) -> Tuple[int, int, int, float]:
    """Lower = sooner. Shipping N=8 INT8 first, then other N=8, then N=4, then 16x16."""
    if n == 8 and cdw == 8:
        tier = 0
    elif n == 8:
        tier = 1
    elif n == 4:
        tier = 2
    else:
        tier = 3
    # Prefer looser periods first (20 before 15 before 12 before 10) so timing
    # fail at 15 ns can skip tighter cells without burning a Vivado hour.
    return (tier, int(cdw), -int(mb), -float(period))


def build_grid(
    *,
    array_sizes: Sequence[int],
    cdws: Sequence[int],
    batches: Sequence[int],
    periods: Sequence[float],
    include_16x16_int4: bool,
) -> List[Tuple[int, int, int, float]]:
    cells: List[Tuple[int, int, int, float]] = []
    for n in array_sizes:
        for cdw in cdws:
            for mb in batches:
                for period in periods:
                    cells.append((int(n), int(cdw), int(mb), float(period)))
    if include_16x16_int4:
        for period in periods:
            cells.append((16, 4, 48, float(period)))
    cells = sorted(set(cells), key=lambda t: priority_rank(*t))
    return cells


def point_info_rank(p: Dict[str, Any]) -> int:
    status = str(p.get("status") or "")
    if status == "closed":
        return 40
    if status.startswith("failed_") and (p.get("failure_detail") or p.get("binding_resource")):
        return 30
    if status.startswith("failed_") or p.get("failure_reason") not in (None, "missing_reports"):
        return 20
    if status == "skipped":
        return 15
    if status == "missing_reports":
        return 5
    return 10


def prefer_point(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    if point_info_rank(new) >= point_info_rank(old):
        return new
    return old


def merge_points(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for p in existing:
        merged[point_key_from_dict(p)] = p
    for p in new:
        key = point_key_from_dict(p)
        if key in merged:
            merged[key] = prefer_point(merged[key], p)
        else:
            merged[key] = p
    return [merged[k] for k in sorted(merged.keys())]


def should_skip_tighter(
    *,
    n: int,
    cdw: int,
    mb: int,
    period: float,
    results_so_far: Dict[Tuple[int, int, int, float], Dict[str, Any]],
) -> Optional[str]:
    """Skip 12/10 if 15 failed timing; skip all periods if LUT/DSP overutil at any period."""
    siblings = [
        (p, res)
        for (nn, cc, mm, p), res in results_so_far.items()
        if nn == n and cc == cdw and mm == mb
    ]
    for _p, res in siblings:
        fr = res.get("failure_reason") or ""
        if "lut_overutilization" in fr or "dsp_overutilization" in fr or "bram_overutilization" in fr:
            return (
                f"Propagated architectural {fr} from sibling period; "
                "util is period-independent on this part."
            )
        status = str(res.get("status") or "")
        if "overutilization" in status:
            return f"Propagated {status} from sibling period."

    if period < 15.0 - 1e-9:
        for p15, res in siblings:
            if abs(p15 - 15.0) < 1e-9 and res.get("status") == "failed_timing":
                return "Skipped tighter than 15 ns because 15 ns failed timing."
    return None


def choose_shipping_point(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer N=8 INT8 mb48 @12 ns (thin but non-marginal); else closed N=8 INT8 @12; else best."""
    preferred = [
        p
        for p in points
        if p.get("status") == "closed"
        and int(p.get("ARRAY_SIZE") or 0) == 8
        and int(p.get("COMPUTE_DATA_WIDTH") or 0) == 8
        and int(p.get("MAX_BATCH_COUNT") or 0) == 48
        and abs(float(p.get("clock_period_ns") or 0) - 12.0) < 1e-9
    ]
    if preferred:
        return preferred[0]
    # Fall back: any closed mb48 N=8 INT8 at 12 ns already checked; try any @12.
    at12 = [
        p
        for p in points
        if p.get("status") == "closed"
        and int(p.get("ARRAY_SIZE") or 0) == 8
        and int(p.get("COMPUTE_DATA_WIDTH") or 0) == 8
        and abs(float(p.get("clock_period_ns") or 0) - 12.0) < 1e-9
    ]
    if at12:
        return max(at12, key=lambda p: int(p.get("MAX_BATCH_COUNT") or 0))
    closed_ship = [
        p
        for p in points
        if p.get("status") == "closed"
        and int(p.get("ARRAY_SIZE") or 0) == 8
        and int(p.get("COMPUTE_DATA_WIDTH") or 0) == 8
        and int(p.get("MAX_BATCH_COUNT") or 0) == 48
    ]
    if closed_ship:
        # Prefer larger period among remaining (more margin) if 12 ns absent.
        return max(closed_ship, key=lambda p: float(p.get("clock_period_ns") or 0.0))
    closed_any = [p for p in points if p.get("status") == "closed"]
    return closed_any[0] if closed_any else None


def choose_demonstrated_ceiling(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """100 MHz / 10 ns close for shipping config, with WNS required inline for any claim."""
    hits = [
        p
        for p in points
        if p.get("status") == "closed"
        and int(p.get("ARRAY_SIZE") or 0) == 8
        and int(p.get("COMPUTE_DATA_WIDTH") or 0) == 8
        and int(p.get("MAX_BATCH_COUNT") or 0) == 48
        and abs(float(p.get("clock_period_ns") or 0) - 10.0) < 1e-9
    ]
    if not hits:
        return None
    p = hits[0]
    wns = (p.get("timing") or {}).get("wns_ns")
    return {
        "label": "demonstrated_100mhz_ceiling",
        "ARRAY_SIZE": 8,
        "COMPUTE_DATA_WIDTH": 8,
        "MAX_BATCH_COUNT": 48,
        "clock_period_ns": 10.0,
        "frequency_mhz_constraint": 100.0,
        "report_prefix": p.get("report_prefix"),
        "wns_ns": wns,
        "margin_class": margin_class_for_wns(None if wns is None else float(wns)),
        "achieved_fmax_mhz_from_period_minus_wns": p.get(
            "achieved_fmax_mhz_from_period_minus_wns"
        ),
        "claim_rule": (
            "Any claim citing 100 MHz must carry the WNS value inline "
            f"(WNS={wns} ns, margin_class="
            f"{margin_class_for_wns(None if wns is None else float(wns))}). "
            "Shipping default remains 12 ns / ~83 MHz."
        ),
    }


def choose_int4_pareto(
    points: List[Dict[str, Any]], throughput_by_config: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    # Prefer throughput row for 16x16 INT4; else any closed INT4 config.
    for row in throughput_by_config:
        if int(row.get("ARRAY_SIZE") or 0) == 16 and int(row.get("COMPUTE_DATA_WIDTH") or 0) == 4:
            return row
    for row in throughput_by_config:
        if int(row.get("COMPUTE_DATA_WIDTH") or 0) == 4:
            return row
    candidates = [
        p
        for p in points
        if p.get("status") == "closed" and int(p.get("COMPUTE_DATA_WIDTH") or 0) == 4
    ]
    n16 = [p for p in candidates if int(p.get("ARRAY_SIZE") or 0) == 16]
    pool = n16 or candidates
    return pool[0] if pool else None


def throughput_for_point(
    p: Optional[Dict[str, Any]], throughput_by_config: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not p:
        return None
    key = config_key(p)
    for row in throughput_by_config:
        if config_key(row) == key:
            return row
    return None


def write_plot(
    points: List[Dict[str, Any]],
    shipping: Optional[Dict[str, Any]],
    int4: Optional[Dict[str, Any]],
    throughput_by_config: List[Dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib unavailable; skipping plot", flush=True)
        return

    closed = [p for p in points if p.get("status") == "closed"]
    failed = [p for p in points if str(p.get("status") or "").startswith("failed_")]
    plot_rows = [r for r in throughput_by_config if r.get("peak_gops") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    for r in plot_rows:
        n = int(r["ARRAY_SIZE"])
        cdw = int(r["COMPUTE_DATA_WIDTH"])
        color = {
            (8, 8): "#1f77b4",
            (8, 4): "#ff7f0e",
            (4, 8): "#2ca02c",
            (4, 4): "#9467bd",
            (16, 4): "#d62728",
        }.get((n, cdw), "#7f7f7f")
        fmax = float(r["achieved_fmax_mhz_from_period_minus_wns"])
        ax.scatter(
            fmax,
            float(r["peak_gops"]),
            c=color,
            marker="o",
            s=42,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            label=f"N={n} INT{cdw} peak",
        )
        ax.scatter(
            fmax,
            float(r["achieved_gops"]),
            c=color,
            marker="x",
            s=36,
            alpha=0.9,
            label=f"N={n} INT{cdw} achieved",
        )
        ax.annotate(
            f"mb{r['MAX_BATCH_COUNT']}@{r['source_clock_period_ns']}ns",
            (fmax, float(r["peak_gops"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="best")
    ax.set_xlabel("Achieved Fmax from tightest close (MHz)")
    ax.set_ylabel("GOP/s")
    ax.set_title("Roofline: one point per config (tightest close)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    acc = load_accuracy_pair()
    xs, ys, labels = [], [], []
    ship_tp = throughput_for_point(shipping, throughput_by_config)
    if ship_tp and ship_tp.get("achieved_gops") is not None:
        xs.append(float(ship_tp["achieved_gops"]))
        ys.append(acc["int8_per_layer_accuracy"] * 100.0)
        labels.append(
            f"ship N={ship_tp['ARRAY_SIZE']} INT{ship_tp['COMPUTE_DATA_WIDTH']} "
            f"mb{ship_tp['MAX_BATCH_COUNT']} @{ship_tp['source_clock_period_ns']}ns"
        )
    if int4 and int4.get("achieved_gops") is not None:
        xs.append(float(int4["achieved_gops"]))
        ys.append(acc["int4_per_layer_accuracy"] * 100.0)
        labels.append(
            f"INT4 N={int4['ARRAY_SIZE']} mb{int4['MAX_BATCH_COUNT']} "
            f"@{int4.get('source_clock_period_ns', int4.get('clock_period_ns'))}ns"
        )
    if xs:
        ax2.plot(xs, ys, "k--", alpha=0.4, linewidth=1)
        for x, y, lab in zip(xs, ys, labels):
            ax2.scatter([x], [y], s=80, zorder=3)
            ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    else:
        ax2.text(
            0.5,
            0.5,
            "No closed Pareto points yet",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
    ax2.set_xlabel("Achieved GOP/s (peak × occupancy)")
    ax2.set_ylabel("Per-layer accuracy (%)")
    ax2.set_title("Accuracy–throughput Pareto")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"uTPU design-space @ PROG_DEPTH={PROG_DEPTH}  "
        f"closed={len(closed)} failed={len(failed)} configs={len(plot_rows)}",
        fontsize=11,
    )
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=140)
    plt.close(fig)
    print(f"Wrote {PLOT}", flush=True)


def write_markdown(
    *,
    points: List[Dict[str, Any]],
    shipping: Optional[Dict[str, Any]],
    int4: Optional[Dict[str, Any]],
    occupancy_meta: Dict[str, Any],
    accuracy: Dict[str, Any],
    skipped: List[Dict[str, Any]],
    throughput_by_config: List[Dict[str, Any]],
    ceiling: Optional[Dict[str, Any]],
    not_yet_attempted: List[Dict[str, Any]],
    sweep_status: str,
) -> None:
    closed = [p for p in points if p.get("status") == "closed"]
    failed = [p for p in points if str(p.get("status") or "").startswith("failed_")]
    ship_tp = throughput_for_point(shipping, throughput_by_config)

    def timing_row(p: Dict[str, Any]) -> str:
        util = p.get("utilization") or {}
        tim = p.get("timing") or {}
        return (
            f"| {p.get('ARRAY_SIZE')} | {p.get('COMPUTE_DATA_WIDTH')} | {p.get('MAX_BATCH_COUNT')} | "
            f"{p.get('clock_period_ns')} | {p.get('status')} | {tim.get('wns_ns')} | "
            f"{p.get('margin_class')} | "
            f"{p.get('achieved_fmax_mhz_from_period_minus_wns')} | "
            f"{util.get('lut_used')}/{util.get('lut_available')} | "
            f"{util.get('dsp_used')}/{util.get('dsp_available')} | "
            f"{util.get('bram_36k_used')}/{util.get('bram_36k_available')} | "
            f"{p.get('binding_resource') or (p.get('failure_reason') or '')} |"
        )

    def tp_row(r: Dict[str, Any]) -> str:
        return (
            f"| {r.get('ARRAY_SIZE')} | {r.get('COMPUTE_DATA_WIDTH')} | {r.get('MAX_BATCH_COUNT')} | "
            f"{r.get('source_clock_period_ns')} | {r.get('source_wns_ns')} | {r.get('margin_class')} | "
            f"{r.get('achieved_fmax_mhz_from_period_minus_wns')} | "
            f"{r.get('peak_gops')} | {r.get('achieved_gops')} |"
        )

    lines: List[str] = []
    lines.append("# Hardware design space (Artix A7-100T)")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()} · git `{git_sha()}` · "
        f"sweep_status=`{sweep_status}`_"
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"Full-top Vivado synth/impl (route, no bitstream) on `{PART}` at "
        f"`PROG_DEPTH={PROG_DEPTH}`, `QUANTIZER_PIPE_DEPTH={PIPE_DEPTH}`, "
        f"`BUFFER_SIZE={BUFFER_SIZE}`."
    )
    lines.append("")
    lines.append(
        "Grid: `ARRAY_SIZE ∈ {4,8}` × `COMPUTE_DATA_WIDTH ∈ {4,8}` × "
        "`MAX_BATCH_COUNT ∈ {4,16,48}` × `period ∈ {20,15,12,10}` ns, plus a "
        "`16×16 INT4` Pareto attempt."
    )
    lines.append("")
    lines.append("## Occupancy (peak ≠ achieved)")
    lines.append("")
    lines.append(f"- **occupancy** = `{occupancy_meta['occupancy']:.6f}`")
    lines.append(f"- **occupancy_source** = {occupancy_meta['occupancy_source']}")
    lines.append(f"- {occupancy_meta.get('occupancy_note', '')}")
    lines.append(
        "- **Throughput** is computed once per `(ARRAY_SIZE, CDW, MAX_BATCH_COUNT)` "
        "from that config's **tightest closing period** (see `throughput_by_config`). "
        "Per-period rows below are timing evidence only."
    )
    lines.append(
        f"- **Fmax derivation**: `{FMAX_DERIVATION['formula']}`. {FMAX_DERIVATION['note']}"
    )
    lines.append("- **peak_gops** = `ARRAY_SIZE^2 * 2 * Fmax_GHz`")
    lines.append("- **achieved_gops** = `peak_gops * occupancy` (never report peak as achieved)")
    lines.append(
        "- **margin_class**: `comfortable` (>1 ns), `thin` (0.2-1 ns), `marginal` (<0.2 ns)."
    )
    lines.append("")
    lines.append("## Shipping point rationale")
    lines.append("")
    if shipping and shipping.get("status") == "closed":
        lines.append(
            f"Chosen shipping point: **N={shipping['ARRAY_SIZE']} INT{shipping['COMPUTE_DATA_WIDTH']} "
            f"MAX_BATCH_COUNT={shipping['MAX_BATCH_COUNT']} @ {shipping['clock_period_ns']} ns "
            f"(~{1000.0/float(shipping['clock_period_ns']):.1f} MHz)** "
            f"(prefix `{shipping.get('report_prefix')}`), "
            f"margin_class=`{shipping.get('margin_class')}`."
        )
        lines.append("")
        lines.append("Why:")
        lines.append(
            f"1. **Accuracy** — INT8 per-layer accuracy "
            f"{accuracy['int8_per_layer_accuracy']*100:.2f}% vs INT4 "
            f"{accuracy['int4_per_layer_accuracy']*100:.2f}% "
            f"(`real_model_accelerator.json`)."
        )
        lines.append(
            "2. **Board fit** — closes on xc7a100t with PROG_DEPTH=65536 instruction BRAM "
            "and remaining LUT/DSP/BRAM headroom for the MNIST/FC class."
        )
        lines.append(
            "3. **Batch ceiling** — MAX_BATCH_COUNT=48 is the largest closing batch from the "
            "timing-closure / LUT bisect path (mb64 LUT-oversubscribes)."
        )
        lines.append(
            "4. **Clock** — **12 ns (~83 MHz)** is the shipping default (WNS thin but "
            "non-marginal). Loose-period closes are floors under met constraints; "
            "**100 MHz is the demonstrated ceiling**, not the shipping default."
        )
        util = shipping.get("utilization") or {}
        tim = shipping.get("timing") or {}
        lines.append("")
        lines.append(
            f"Evidence: WNS={tim.get('wns_ns')} ns ({shipping.get('margin_class')}), "
            f"constraint Fmax={1000.0/float(shipping['clock_period_ns']):.2f} MHz, "
            f"period-WNS Fmax≈{shipping.get('achieved_fmax_mhz_from_period_minus_wns')} MHz, "
            f"LUT={util.get('lut_used')}/{util.get('lut_available')}, "
            f"DSP={util.get('dsp_used')}/{util.get('dsp_available')}, "
            f"BRAM={util.get('bram_36k_used')}/{util.get('bram_36k_available')}."
        )
        if ship_tp:
            lines.append(
                f"Config throughput (from tightest close "
                f"@{ship_tp.get('source_clock_period_ns')} ns, "
                f"WNS={ship_tp.get('source_wns_ns')}): "
                f"peak={ship_tp.get('peak_gops')} GOP/s, "
                f"achieved={ship_tp.get('achieved_gops')} GOP/s."
            )
        if shipping.get("binding_resource"):
            lines.append(
                f"Highest util resource at shipping close: **{shipping['binding_resource']}** "
                f"(not oversubscribed)."
            )
    else:
        lines.append("Shipping point not yet closed in this artifact — see skipped/failed table.")
    lines.append("")
    if ceiling:
        lines.append("## Demonstrated 100 MHz ceiling")
        lines.append("")
        lines.append(
            f"- Constraint **100 MHz** (10 ns) closed with **WNS={ceiling.get('wns_ns')} ns** "
            f"(`{ceiling.get('margin_class')}`)."
        )
        lines.append(f"- {ceiling.get('claim_rule')}")
        lines.append("")
    lines.append("## Accuracy–throughput Pareto")
    lines.append("")
    if ship_tp and ship_tp.get("achieved_gops") is not None:
        lines.append(
            f"- Point A (INT8): accuracy={accuracy['int8_per_layer_accuracy']*100:.2f}%, "
            f"achieved_gops={ship_tp.get('achieved_gops')} "
            f"(tightest close @{ship_tp.get('source_clock_period_ns')} ns, "
            f"WNS={ship_tp.get('source_wns_ns')})"
        )
    if int4 and int4.get("achieved_gops") is not None:
        lines.append(
            f"- Point B (INT4 N={int4.get('ARRAY_SIZE')}): "
            f"accuracy={accuracy['int4_per_layer_accuracy']*100:.2f}%, "
            f"achieved_gops={int4.get('achieved_gops')} "
            f"(prefix `{int4.get('source_report_prefix') or int4.get('report_prefix')}`)"
        )
    else:
        lines.append(
            "- Point B (16x16 INT4): **did not close** or not yet attempted — "
            "see not_yet_attempted / timing table."
        )
    lines.append("")
    lines.append("## Throughput by config (one row per NxCDWxMB)")
    lines.append("")
    lines.append(
        "| N | CDW | MB | source_period_ns | WNS | margin | Fmax_MHz | peak_GOP/s | achieved_GOP/s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for r in throughput_by_config:
        lines.append(tp_row(r))
    lines.append("")
    lines.append("## Timing evidence by corner (per period)")
    lines.append("")
    lines.append(
        "| N | CDW | MB | period_ns | status | WNS | margin | Fmax_MHz | LUT | DSP | BRAM | binder |"
    )
    lines.append("|---:|---:|---:|---:|---|---:|---|---:|---|---|---|---|")
    for p in sorted(
        points,
        key=lambda x: (
            int(x.get("ARRAY_SIZE") or 0),
            int(x.get("COMPUTE_DATA_WIDTH") or 0),
            int(x.get("MAX_BATCH_COUNT") or 0),
            -float(x.get("clock_period_ns") or 0),
        ),
    ):
        lines.append(timing_row(p))
    lines.append("")
    lines.append("## Summary counts")
    lines.append("")
    lines.append(f"- sweep_status: **{sweep_status}**")
    lines.append(f"- closed (attempted): **{len(closed)}**")
    lines.append(f"- failed (attempted): **{len(failed)}**")
    lines.append(f"- not_yet_attempted: **{len(not_yet_attempted)}**")
    if skipped:
        lines.append("")
        lines.append("### Explicitly skipped (honest)")
        lines.append("")
        for s in skipped:
            lines.append(
                f"- `{s.get('report_prefix') or s.get('key')}`: {s.get('skip_reason')}"
            )
    lines.append("")
    if not_yet_attempted:
        lines.append("## Not yet attempted")
        lines.append("")
        lines.append("| N | CDW | MB | period_ns | canonical_prefix |")
        lines.append("|---:|---:|---:|---:|---|")
        for c in not_yet_attempted:
            lines.append(
                f"| {c['ARRAY_SIZE']} | {c['COMPUTE_DATA_WIDTH']} | {c['MAX_BATCH_COUNT']} | "
                f"{c['clock_period_ns']} | `{c['canonical_prefix']}` |"
            )
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- JSON: `bench/results/design_space_sweep.json`")
    lines.append("- Plot: `docs/design_space_roofline.png`")
    lines.append("- Runner: `firmware/host/run_design_space_sweep.py`")
    lines.append("- TCL: `scripts/synth_design_space_point.tcl`")
    lines.append("")
    DOCS.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {MD}", flush=True)


def load_existing_points() -> List[Dict[str, Any]]:
    if not OUT.exists():
        return []
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    pts = data.get("points")
    return list(pts) if isinstance(pts, list) else []


def emit_artifact(
    points: List[Dict[str, Any]],
    *,
    occupancy_meta: Dict[str, Any],
    accuracy: Dict[str, Any],
    skipped: List[Dict[str, Any]],
    attempted_this_run: List[str],
    reused: List[str],
    full_grid: List[Tuple[int, int, int, float]],
) -> Dict[str, Any]:
    occupancy = float(occupancy_meta["occupancy"])
    attempted_points = [p for p in points if point_was_attempted(p)]
    attempted_keys = {
        (
            int(p.get("ARRAY_SIZE") or 0),
            int(p.get("COMPUTE_DATA_WIDTH") or 0),
            int(p.get("MAX_BATCH_COUNT") or 0),
            float(p.get("clock_period_ns") or 0.0),
        )
        for p in attempted_points
    }
    not_yet_attempted = [
        grid_cell_dict(n, cdw, mb, period)
        for (n, cdw, mb, period) in full_grid
        if (int(n), int(cdw), int(mb), float(period)) not in attempted_keys
    ]
    not_yet_attempted = sorted(
        not_yet_attempted,
        key=lambda c: (
            c["ARRAY_SIZE"],
            c["COMPUTE_DATA_WIDTH"],
            -c["MAX_BATCH_COUNT"],
            -c["clock_period_ns"],
        ),
    )

    throughput_by_config = build_throughput_by_config(attempted_points, occupancy)
    shipping = choose_shipping_point(attempted_points)
    ceiling = choose_demonstrated_ceiling(attempted_points)
    int4 = choose_int4_pareto(attempted_points, throughput_by_config)
    closed = [p for p in attempted_points if p.get("status") == "closed"]
    failed = [p for p in attempted_points if str(p.get("status") or "").startswith("failed_")]
    sweep_status = "complete" if not not_yet_attempted else "partial"

    ship_out = dict(shipping) if shipping else None
    if ship_out is not None:
        tp = throughput_for_point(shipping, throughput_by_config)
        if tp:
            ship_out["throughput_from_tightest_close"] = {
                "source_clock_period_ns": tp.get("source_clock_period_ns"),
                "source_wns_ns": tp.get("source_wns_ns"),
                "peak_gops": tp.get("peak_gops"),
                "achieved_gops": tp.get("achieved_gops"),
                "margin_class": tp.get("margin_class"),
            }

    artifact = {
        "schema_version": 2,
        "status": sweep_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "part": PART,
        "fixed_generics": {
            "PROG_DEPTH": PROG_DEPTH,
            "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
            "BUFFER_SIZE": BUFFER_SIZE,
            "EXT_ADDR_EN": EXT_ADDR_EN,
        },
        "methodology": {
            "flow": (
                "Full-top synth/impl via scripts/synth_design_space_point.tcl "
                "(route_design, no bitstream). Reuses matching build/reports prefixes "
                "including prog_depth_pd65536 alias for N=8 INT8 mb48 @20ns. "
                "Priority: N=8 INT8 mb{4,16,48} periods first. "
                "Skip tighter clocks after 15 ns timing fail; propagate LUT/DSP/BRAM "
                "overutilization across periods."
            ),
            "timing_vs_throughput": (
                "Per-period points[] rows are timing evidence (period, WNS, util, "
                "margin_class). Plotted peak/achieved GOP/s are computed once per "
                "(ARRAY_SIZE, COMPUTE_DATA_WIDTH, MAX_BATCH_COUNT) in "
                "throughput_by_config, using that config's tightest closing period."
            ),
            "fmax_derivation": dict(FMAX_DERIVATION),
            "margin_class": {
                "comfortable": ">1.0 ns WNS",
                "thin": "0.2-1.0 ns WNS",
                "marginal": "<0.2 ns WNS",
            },
            "shipping_default": "N=8 INT8 MAX_BATCH_COUNT=48 @ 12 ns (~83 MHz)",
            "demonstrated_ceiling_rule": (
                "100 MHz (10 ns) closes are recorded as demonstrated ceiling; "
                "any 100 MHz claim must quote WNS inline. Not the shipping default."
            ),
            "seed_repro_pending": (
                "If the grid finishes with time to spare, re-run mb48 @10 ns under "
                "2-3 alternate implementation seeds and record WNS spread."
            ),
        },
        "occupancy": occupancy_meta,
        "accuracy_pareto_reference": accuracy,
        "points": attempted_points,
        "throughput_by_config": throughput_by_config,
        "shipping_point": ship_out,
        "demonstrated_fmax_ceiling": ceiling,
        "int4_pareto_point": int4,
        "not_yet_attempted": not_yet_attempted,
        "summary": {
            "status": sweep_status,
            "n_attempted": len(attempted_points),
            "n_closed": len(closed),
            "n_failed": len(failed),
            "n_not_yet_attempted": len(not_yet_attempted),
            "n_throughput_configs": len(throughput_by_config),
            "reused_report_prefixes": reused,
            "vivado_attempted_this_run": attempted_this_run,
            "skipped": skipped,
        },
        "plot_path": str(PLOT.relative_to(REPO)).replace("\\", "/"),
        "doc_path": str(MD.relative_to(REPO)).replace("\\", "/"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_plot(attempted_points, shipping, int4, throughput_by_config)
    write_markdown(
        points=attempted_points,
        shipping=shipping,
        int4=int4,
        occupancy_meta=occupancy_meta,
        accuracy=accuracy,
        skipped=skipped,
        throughput_by_config=throughput_by_config,
        ceiling=ceiling,
        not_yet_attempted=not_yet_attempted,
        sweep_status=sweep_status,
    )
    print(
        f"Wrote {OUT} status={sweep_status} attempted={len(attempted_points)} "
        f"closed={len(closed)} failed={len(failed)} not_yet={len(not_yet_attempted)}",
        flush=True,
    )
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="Launch Vivado for missing points")
    ap.add_argument("--collect-only", action="store_true", help="Only harvest existing reports")
    ap.add_argument(
        "--priority-only",
        action="store_true",
        help="Only N=8 INT8 mb{4,16,48} x periods (shipping-relevant subset)",
    )
    ap.add_argument("--include-16x16-int4", action="store_true", default=True)
    ap.add_argument("--no-16x16-int4", action="store_true")
    ap.add_argument("--array-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--cdws", type=int, nargs="+", default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=None)
    ap.add_argument("--periods", type=float, nargs="+", default=None)
    ap.add_argument("--max-vivado", type=int, default=0, help="Cap Vivado launches (0=unlimited)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--force-prefix",
        type=str,
        default=None,
        help="Run/collect a single canonical prefix (debug)",
    )
    args = ap.parse_args()

    do_run = bool(args.run) and not args.collect_only
    include_16 = bool(args.include_16x16_int4) and not bool(args.no_16x16_int4)

    if args.priority_only:
        array_sizes, cdws, batches = [8], [8], [4, 16, 48]
        include_16 = False
    else:
        array_sizes = args.array_sizes or ARRAY_SIZES
        cdws = args.cdws or CDWS
        batches = args.batches or BATCHES
    periods = args.periods or PERIODS

    occupancy_meta = load_occupancy()
    occupancy = float(occupancy_meta["occupancy"])
    accuracy = load_accuracy_pair()

    existing = load_existing_points()
    existing_map = {point_key_from_dict(p): p for p in existing}

    if args.force_prefix:
        # Debug single point: parse from prefix when possible.
        m = re.match(
            r"dss_n(\d+)_cdw(\d+)_mb(\d+)_clk([\d.]+)_pd(\d+)_prog(\d+)",
            args.force_prefix,
        )
        if m:
            grid = [(int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))]
        else:
            print("ERROR: --force-prefix must be canonical dss_n*_cdw*_mb*_clk* form", flush=True)
            return 2
    else:
        grid = build_grid(
            array_sizes=array_sizes,
            cdws=cdws,
            batches=batches,
            periods=periods,
            include_16x16_int4=include_16,
        )

    new_points: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    attempted: List[str] = []
    reused: List[str] = []
    results_so_far: Dict[Tuple[int, int, int, float], Dict[str, Any]] = {}

    # Seed results_so_far from existing richer points.
    for p in existing:
        results_so_far[
            (
                int(p.get("ARRAY_SIZE") or 0),
                int(p.get("COMPUTE_DATA_WIDTH") or 0),
                int(p.get("MAX_BATCH_COUNT") or 0),
                float(p.get("clock_period_ns") or 0.0),
            )
        ] = p

    vivado_launches = 0
    for n, cdw, mb, period in grid:
        prefix = resolve_prefix(n, cdw, mb, period)
        key = (n, cdw, mb, period)

        skip_reason = should_skip_tighter(
            n=n, cdw=cdw, mb=mb, period=period, results_so_far=results_so_far
        )
        if skip_reason and not reports_present(prefix):
            stub = {
                "report_prefix": report_prefix(n, cdw, mb, period),
                "ARRAY_SIZE": n,
                "COMPUTE_DATA_WIDTH": cdw,
                "ACCUMULATOR_DATA_WIDTH": accum_width(cdw),
                "MAX_BATCH_COUNT": mb,
                "clock_period_ns": period,
                "frequency_mhz_constraint": 1000.0 / period,
                "PROG_DEPTH": PROG_DEPTH,
                "BUFFER_SIZE": BUFFER_SIZE,
                "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
                "status": "skipped",
                "skip_reason": skip_reason,
                "failure_reason": skip_reason.split()[1] if "overutilization" in skip_reason else "skipped_heuristic",
                "binding_resource": (
                    "lut"
                    if "lut_" in skip_reason
                    else "dsp"
                    if "dsp_" in skip_reason
                    else "bram"
                    if "bram_" in skip_reason
                    else "timing"
                    if "timing" in skip_reason
                    else None
                ),
            }
            # Copy failure_detail from sibling if present.
            for (_p, res) in list(results_so_far.items()):
                if (
                    _p[0] == n
                    and _p[1] == cdw
                    and _p[2] == mb
                    and res.get("failure_detail")
                    and "overutilization" in str(res.get("failure_reason") or res.get("status") or "")
                ):
                    stub["failure_detail"] = dict(res["failure_detail"])
                    stub["failure_reason"] = res.get("failure_reason") or stub["failure_reason"]
                    stub["binding_resource"] = res.get("binding_resource") or stub["binding_resource"]
                    stub["status"] = f"failed_{stub['failure_reason']}" if stub.get("failure_reason") and "overutilization" in str(stub.get("failure_reason")) else "skipped"
                    break
            stub = attach_throughput(stub, occupancy)
            new_points.append(stub)
            results_so_far[key] = stub
            skipped.append({"key": list(key), "report_prefix": stub["report_prefix"], "skip_reason": skip_reason})
            continue

        if reports_present(prefix):
            reused.append(prefix)
            pt = collect_point(n=n, cdw=cdw, mb=mb, period=period, occupancy=occupancy, prefix=prefix)
            new_points.append(pt)
            results_so_far[key] = pt
            continue

        # Also accept canonical prefix if alias was preferred but missing.
        canonical = report_prefix(n, cdw, mb, period)
        if prefix != canonical and reports_present(canonical):
            reused.append(canonical)
            pt = collect_point(n=n, cdw=cdw, mb=mb, period=period, occupancy=occupancy, prefix=canonical)
            new_points.append(pt)
            results_so_far[key] = pt
            continue

        if do_run:
            if args.max_vivado and vivado_launches >= args.max_vivado:
                stub = {
                    "report_prefix": canonical,
                    "ARRAY_SIZE": n,
                    "COMPUTE_DATA_WIDTH": cdw,
                    "ACCUMULATOR_DATA_WIDTH": accum_width(cdw),
                    "MAX_BATCH_COUNT": mb,
                    "clock_period_ns": period,
                    "PROG_DEPTH": PROG_DEPTH,
                    "BUFFER_SIZE": BUFFER_SIZE,
                    "QUANTIZER_PIPE_DEPTH": PIPE_DEPTH,
                    "status": "skipped",
                    "skip_reason": f"max-vivado cap ({args.max_vivado}) reached",
                }
                stub = attach_throughput(stub, occupancy)
                new_points.append(stub)
                results_so_far[key] = stub
                skipped.append(
                    {
                        "key": list(key),
                        "report_prefix": canonical,
                        "skip_reason": stub["skip_reason"],
                    }
                )
                continue
            rc = run_vivado(n=n, cdw=cdw, mb=mb, period=period, prefix=canonical, jobs=args.jobs)
            vivado_launches += 1
            attempted.append(canonical)
            if rc != 0:
                print(f"WARN vivado rc={rc} for {canonical}", flush=True)
            pt = collect_point(n=n, cdw=cdw, mb=mb, period=period, occupancy=occupancy, prefix=canonical)
            new_points.append(pt)
            results_so_far[key] = pt
            # Checkpoint artifact after each Vivado point so a kill mid-sweep keeps progress.
            merged = merge_points(existing, new_points)
            emit_artifact(
                merged,
                occupancy_meta=occupancy_meta,
                accuracy=accuracy,
                skipped=skipped,
                attempted_this_run=attempted,
                reused=reused,
                full_grid=grid,
            )
            existing = merged
            continue

        # collect-only / no --run: keep prior evidence; do not invent stubs for never-attempted.
        if key in existing_map:
            pt = attach_throughput(dict(existing_map[key]), occupancy)
            # Refresh from disk if reports appeared.
            if reports_present(pt.get("report_prefix") or canonical):
                pt = collect_point(
                    n=n,
                    cdw=cdw,
                    mb=mb,
                    period=period,
                    occupancy=occupancy,
                    prefix=str(pt.get("report_prefix") or canonical),
                )
            new_points.append(pt)
            results_so_far[key] = pt
        elif reports_present(canonical):
            pt = collect_point(n=n, cdw=cdw, mb=mb, period=period, occupancy=occupancy, prefix=canonical)
            new_points.append(pt)
            results_so_far[key] = pt
        # else: leave as gap (not recorded) unless --run

    points = merge_points(existing, new_points)
    # Refresh throughput on all points with current occupancy.
    points = [attach_throughput(dict(p), occupancy) for p in points]
    artifact = emit_artifact(
        points,
        occupancy_meta=occupancy_meta,
        accuracy=accuracy,
        skipped=skipped,
        attempted_this_run=attempted,
        reused=reused,
        full_grid=grid,
    )
    summary = artifact["summary"]
    print(
        "SUMMARY",
        f"status={summary.get('status')}",
        f"closed={summary['n_closed']}",
        f"failed={summary['n_failed']}",
        f"not_yet={summary.get('n_not_yet_attempted')}",
        f"reused={len(reused)}",
        f"vivado={len(attempted)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
