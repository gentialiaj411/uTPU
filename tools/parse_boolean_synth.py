"""Parse genuine Vivado reports for the Boolean Spartan-7 Phase-3 gate."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "build" / "reports"
OUT = ROOT / "bench" / "results" / "boolean_vec_unit_synth.json"
PART = "xc7s50csga324-1"


def read_report(prefix: str) -> dict:
    util = (REPORTS / f"{prefix}_utilization.rpt").read_text(encoding="utf-8", errors="replace")
    timing = (REPORTS / f"{prefix}_timing_summary.rpt").read_text(encoding="utf-8", errors="replace")
    device = re.search(r"(?m)^\|\s*Device\s*:\s*(\S+)", util)
    design = re.search(r"(?m)^\|\s*Design\s*:\s*(\S+)", util)
    if not device or not design or design.group(1) != "top_board":
        raise ValueError(f"{prefix}: device/top identity missing or unexpected")
    if "Design State : Routed" not in util:
        raise ValueError(f"{prefix}: utilization report is not from a routed design")

    def first(pattern: str, source: str, label: str) -> float:
        match = re.search(pattern, source, re.MULTILINE)
        if not match:
            raise ValueError(f"{prefix}: could not parse {label}")
        return float(match.group(1).replace(",", ""))

    lut = first(r"^\|\s*Slice LUTs\*?\s*\|\s*([\d,]+)", util, "LUT")
    ff = first(r"^\|\s*Slice Registers\s*\|\s*([\d,]+)", util, "FF")
    bram = first(r"^\|\s*Block RAM Tile\s*\|\s*([\d.]+)", util, "BRAM")
    dsp = first(r"^\|\s*DSPs\s*\|\s*([\d,]+)", util, "DSP")
    clocks = re.findall(r"(?m)^\s*([^\s]+)\s+\{[^}]+\}\s+([\d.]+)\s+[\d.]+", timing)
    core_clock = next((name for name, period in clocks if abs(float(period) - 20.0) < 0.05), None)
    if core_clock is None:
        raise ValueError(f"{prefix}: no 20 ns derived core clock in timing report: {clocks}")
    timing_row = re.search(rf"(?m)^\s*{re.escape(core_clock)}\s+(-?[\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+(-?[\d.]+)", timing)
    if not timing_row:
        raise ValueError(f"{prefix}: could not parse WNS/WHS for {core_clock}")
    # The first max-delay path in the summary is the worst setup path.
    from_to = re.search(r"(?m)^\s*Startpoint:\s*(.+?)\s*$.*?^\s*Endpoint:\s*(.+?)\s*$", timing, re.DOTALL)
    if not from_to:
        from_to = re.search(r"(?m)^\s*Source:\s*(.+?)\s*$.*?^\s*Destination:\s*(.+?)\s*$", timing, re.DOTALL)
    return {"lut": int(lut), "ff": int(ff), "bram": bram, "dsp": int(dsp),
            "part": "xc7s50csga324-1" if device.group(1) == "xc7s50csga324-1" else device.group(1),
            "top": design.group(1), "core_clock_name": core_clock,
            "wns_ns": float(timing_row.group(1)), "whs_ns": float(timing_row.group(2)),
            "worst_path": {"startpoint": from_to.group(1).strip() if from_to else None,
                           "endpoint": from_to.group(2).strip() if from_to else None},
            "timing_report": f"build/reports/{prefix}_timing_summary.rpt",
            "utilization_report": f"build/reports/{prefix}_utilization.rpt"}


def main() -> int:
    for mode in ("0", "1"):
        log = (ROOT / "build" / f"boolean_vec{mode}_vivado.log").read_text(encoding="utf-8", errors="replace")
        if "impl_1 PROGRESS=100% STATUS=route_design Complete!" not in log or "DSS_DONE" not in log:
            raise ValueError(f"VEC={mode} full implementation did not complete")
    vec0 = read_report("boolean_vec0")
    vec1 = read_report("boolean_vec1")
    ooc_timing = (REPORTS / "vector_ooc_timing.rpt").read_text(encoding="utf-8", errors="replace")
    ooc = re.search(r"(?m)^\s*(-?[\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+(-?[\d.]+)", ooc_timing)
    if not ooc:
        raise ValueError("vector OOC WNS/WHS missing")
    if vec0["part"] != vec1["part"]:
        raise ValueError("VEC=0 and VEC=1 reports target different parts")
    payload = {"status": "ok", "part": vec1["part"], "top": "top_board",
               "clock_period_pin_ns": 10.0, "core_clock_hz": round(1e9 / 20.0),
               "vector_unit_ooc": {"clock_period_ns": 20.0, "wns_ns": float(ooc.group(1)),
                                    "whs_ns": float(ooc.group(2)),
                                    "report": "build/reports/vector_ooc_timing.rpt"},
               "modes": {"0": vec0, "1": vec1},
               "delta": {key: vec1[key] - vec0[key] for key in ("lut", "ff", "bram", "dsp")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
