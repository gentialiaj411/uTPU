from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "packed_dsp_synth.json"
REPORTS_DIR = REPO_ROOT / "build" / "reports"

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name": "packed_baseline_8x8_int8",
        "report_prefix": "packed_baseline_8x8_int8",
        "top_name": "top",
        "params": {
            "ARRAY_SIZE": 8,
            "COMPUTE_DATA_WIDTH": 8,
            "ACCUMULATOR_DATA_WIDTH": 32,
            "BUFFER_SIZE": 4096,
            "EXT_ADDR_EN": 1,
        },
        "hypothesis_dsp": 64,
    },
    {
        "name": "packed_baseline_16x16_int8",
        "report_prefix": "packed_baseline_16x16_int8",
        "top_name": "top",
        "params": {
            "ARRAY_SIZE": 16,
            "COMPUTE_DATA_WIDTH": 8,
            "ACCUMULATOR_DATA_WIDTH": 32,
            "BUFFER_SIZE": 4096,
            "EXT_ADDR_EN": 1,
        },
        "hypothesis_dsp": 256,
    },
    {
        "name": "packed_array_8x8_int8",
        "report_prefix": "packed_array_8x8_int8",
        "top_name": "top_packed",
        "params": {
            "ARRAY_SIZE": 8,
            "COMPUTE_DATA_WIDTH": 8,
            "ACCUMULATOR_DATA_WIDTH": 32,
        },
        "hypothesis_dsp": 32,
    },
    {
        "name": "packed_array_16x16_int8",
        "report_prefix": "packed_array_16x16_int8",
        "top_name": "top_packed",
        "params": {
            "ARRAY_SIZE": 16,
            "COMPUTE_DATA_WIDTH": 8,
            "ACCUMULATOR_DATA_WIDTH": 32,
        },
        "hypothesis_dsp": 128,
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _parse_timing(text: str) -> Dict[str, Any]:
    wns = whs = tns = ths = None
    summary_re = re.search(
        r"^\s*([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+\s+([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+",
        text,
        re.MULTILINE,
    )
    if summary_re:
        wns = float(summary_re.group(1))
        tns = float(summary_re.group(2))
        whs = float(summary_re.group(3))
        ths = float(summary_re.group(4))
    return {
        "wns_ns": wns,
        "whs_ns": whs,
        "tns_ns": tns if tns is not None else 0.0,
        "ths_ns": ths if ths is not None else 0.0,
        "all_paths_met": (wns is not None and wns >= 0.0 and (whs is None or whs >= 0.0)),
    }


def _parse_utilization(text: str) -> Dict[str, Optional[int]]:
    def grab_used_available(label: str) -> tuple[Optional[int], Optional[int]]:
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+)\s*\|.*?\|\s*(\d+)\s*\|", text)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    lut_used, lut_available = grab_used_available("LUT as Logic")
    ff_used, ff_available = grab_used_available("Slice Registers")
    bram_used, bram_available = grab_used_available("Block RAM Tile")
    dsp_used, dsp_available = grab_used_available("DSPs")

    return {
        "lut_used": lut_used,
        "lut_available": lut_available,
        "ff_used": ff_used,
        "ff_available": ff_available,
        "bram_36k_used": bram_used,
        "bram_36k_available": bram_available,
        "dsp_used": dsp_used,
        "dsp_available": dsp_available,
    }


def _parse_route_status(text: str) -> str:
    lowered = text.lower()
    if "route design completed successfully" in lowered or "design is fully routed" in lowered:
        return "clean"
    if "write_bitstream complete" in lowered or "impl_status: write bitstream complete" in lowered:
        return "clean"
    if "unrouted" in lowered or "route_design failed" in lowered:
        return "unrouted"
    if "place_design failed" in lowered or "place failed" in lowered:
        return "unplaced"
    if "error" in lowered or "failed" in lowered:
        return "error"
    return "unknown"


def _parse_vivado_version(*texts: str) -> Optional[str]:
    for text in texts:
        m = re.search(r"Tool Version\s*:\s*Vivado v\.([0-9.]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"vivado v([0-9.]+)", text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _build_run(spec: Dict[str, Any]) -> Dict[str, Any]:
    prefix = spec["report_prefix"]
    timing_path = REPORTS_DIR / f"{prefix}_timing_summary.rpt"
    util_path = REPORTS_DIR / f"{prefix}_utilization.rpt"
    route_path = REPORTS_DIR / f"{prefix}_route_status.rpt"
    synth_util_path = REPORTS_DIR / f"{prefix}_synth_utilization.rpt"
    report_files = [
        f"build/reports/{prefix}_timing_summary.rpt",
        f"build/reports/{prefix}_utilization.rpt",
        f"build/reports/{prefix}_route_status.rpt",
        f"build/reports/{prefix}.bit",
    ]
    run: Dict[str, Any] = {
        "name": spec["name"],
        "params": spec["params"],
        "part": "xc7a100tcsg324-1",
        "vivado_version": None,
        "timing": None,
        "utilization": None,
        "route_status": None,
        "report_files": report_files,
    }
    util_source = None
    if util_path.is_file():
        util_source = util_path
    elif synth_util_path.is_file():
        util_source = synth_util_path

    timing_text = timing_path.read_text(encoding="utf-8", errors="replace") if timing_path.is_file() else ""
    util_text = util_source.read_text(encoding="utf-8", errors="replace") if util_source else ""
    route_text = route_path.read_text(encoding="utf-8", errors="replace") if route_path.is_file() else ""

    if util_text:
        run["utilization"] = _parse_utilization(util_text)
    if timing_text:
        run["timing"] = _parse_timing(timing_text)
    if route_text:
        run["route_status"] = _parse_route_status(route_text)
    elif util_text:
        run["route_status"] = "missing"
    run["vivado_version"] = _parse_vivado_version(timing_text, util_text, route_text)
    return run


def build_artifact() -> Dict[str, Any]:
    runs = [_build_run(spec) for spec in RUN_SPECS]
    return {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write packed_dsp_synth.json from Vivado reports.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()
    artifact = build_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    parsed_runs = sum(1 for run in artifact["runs"] if run["utilization"] is not None or run["timing"] is not None)
    print(f"[write_packed_dsp_synth_json] wrote {args.output} runs_with_reports={parsed_runs}/{len(artifact['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
